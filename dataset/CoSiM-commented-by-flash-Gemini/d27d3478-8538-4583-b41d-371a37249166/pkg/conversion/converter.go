/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file converter.go
 * @brief Generic reflection-driven type translation engine with path-aware diagnostics.
 * 
 * Functional Intent: Facilitates deep transformation of complex data structures, 
 * typically used for migrating between different API schema versions. It supports 
 * both automatic reflection-based field matching and specialized expert handlers. 
 * A key feature of this implementation is the dual-stack path tracking (src/dest), 
 * which provides high-fidelity diagnostic information identifying the exact 
 * nested field where a conversion failure occurred.
 * 
 * Domain: Production Systems, Schema Evolution, Kubernetes API Machinery.
 */

package conversion

import (
	"fmt"
	"reflect"
)

type typePair struct {
	source reflect.Type
	dest   reflect.Type
}

type typeNamePair struct {
	fieldType reflect.Type
	fieldName string
}

/**
 * @interface DebugLogger
 * @brief Diagnostic interface for tracing the internal decision tree of the converter.
 */
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

/**
 * @struct Converter
 * @brief Central registry for conversion rules and expert logic.
 */
type Converter struct {
	// funcs - Map of type pairs to specialized transformation functions.
	funcs map[typePair]reflect.Value

	// structFieldDests - Registry of explicit source-to-destination field overrides.
	structFieldDests map[typeNamePair][]typeNamePair

	// structFieldSources - Registry of explicit destination-from-source field overrides.
	structFieldSources map[typeNamePair][]typeNamePair

	Debug DebugLogger

	// NameFunc - Discriminator for determining type identity compatibility.
	NameFunc func(t reflect.Type) string
}

/**
 * NewConverter - Initializes a fresh conversion context with default type naming logic.
 */
func NewConverter() *Converter {
	return &Converter{
		funcs:              map[typePair]reflect.Value{},
		NameFunc:           func(t reflect.Type) string { return t.Name() },
		structFieldDests:   map[typeNamePair][]typeNamePair{},
		structFieldSources: map[typeNamePair][]typeNamePair{},
	}
}

/**
 * @interface Scope
 * @brief Contextual handle passed to recursive conversion operations.
 */
type Scope interface {
	Convert(src, dest interface{}, flags FieldMatchingFlags) error

	SrcTag() reflect.StructTag
	DestTag() reflect.StructTag

	Flags() FieldMatchingFlags

	Meta() *Meta
}

/**
 * @struct Meta
 * @brief High-level metadata (e.g. API versions) associated with the current conversion session.
 */
type Meta struct {
	SrcVersion  string
	DestVersion string
}

/**
 * @struct scope
 * @brief Internal implementation of Scope, maintaining synchronized path stacks for error reporting.
 */
type scope struct {
	converter *Converter
	meta      *Meta
	flags     FieldMatchingFlags

	// srcStack & destStack - Dual stacks to track the current location in the object hierarchy.
	srcStack  scopeStack
	destStack scopeStack
}

type scopeStackElem struct {
	tag   reflect.StructTag
	value reflect.Value
	key   string
}

type scopeStack []scopeStackElem

func (s *scopeStack) pop() {
	n := len(*s)
	*s = (*s)[:n-1]
}

func (s *scopeStack) push(e scopeStackElem) {
	*s = append(*s, e)
}

func (s *scopeStack) top() *scopeStackElem {
	return &(*s)[len(*s)-1]
}

/**
 * describe - Generates a string representation of the current object path (e.g. ".Spec.Template.Name").
 */
func (s scopeStack) describe() string {
	desc := ""
	if len(s) > 1 {
		desc = "(" + s[1].value.Type().String() + ")"
	}
	for i, v := range s {
		if i < 2 {
			continue
		}
		if v.key == "" {
			desc += fmt.Sprintf(".%v", v.value.Type())
		} else {
			desc += fmt.Sprintf(".%v", v.key)
		}
	}
	return desc
}

// setIndices - Contextual markers for slice/array iteration.
func (s *scope) setIndices(src, dest int) {
	s.srcStack.top().key = fmt.Sprintf("[%v]", src)
	s.destStack.top().key = fmt.Sprintf("[%v]", dest)
}

// setKeys - Contextual markers for map iteration.
func (s *scope) setKeys(src, dest interface{}) {
	s.srcStack.top().key = fmt.Sprintf(`["%v"]`, src)
	s.destStack.top().key = fmt.Sprintf(`["%v"]`, dest)
}

func (s *scope) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.Convert(src, dest, flags, s.meta)
}

func (s *scope) SrcTag() reflect.StructTag {
	return s.srcStack.top().tag
}

func (s *scope) DestTag() reflect.StructTag {
	return s.destStack.top().tag
}

func (s *scope) Flags() FieldMatchingFlags {
	return s.flags
}

func (s *scope) Meta() *Meta {
	return s.meta
}

func (s *scope) describe() (src, dest string) {
	return s.srcStack.describe(), s.destStack.describe()
}

/**
 * error - Builder for high-signal error messages including the source and destination paths.
 */
func (s *scope) error(message string, args ...interface{}) error {
	srcPath, destPath := s.describe()
	where := fmt.Sprintf("converting %v to %v: ", srcPath, destPath)
	return fmt.Errorf(where+message, args...)
}

/**
 * Register - Adds a specialized conversion function for a specific type pair.
 * 
 * Logic: Validates that the provided function conforms to the 
 * (InPtr, OutPtr, Scope) -> error signature required for dynamic dispatch.
 */
func (c *Converter) Register(conversionFunc interface{}) error {
	fv := reflect.ValueOf(conversionFunc)
	ft := fv.Type()
	if ft.Kind() != reflect.Func {
		return fmt.Errorf("expected func, got: %v", ft)
	}
	if ft.NumIn() != 3 {
		return fmt.Errorf("expected three 'in' params, got: %v", ft)
	}
	if ft.NumOut() != 1 {
		return fmt.Errorf("expected one 'out' param, got: %v", ft)
	}
	if ft.In(0).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 0, got: %v", ft)
	}
	if ft.In(1).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 1, got: %v", ft)
	}
	scopeType := Scope(nil)
	if e, a := reflect.TypeOf(&scopeType).Elem(), ft.In(2); e != a {
		return fmt.Errorf("expected '%v' arg for 'in' param 2, got '%v' (%v)", e, a, ft)
	}
	var forErrorType error
	errorType := reflect.TypeOf(&forErrorType).Elem()
	if ft.Out(0) != errorType {
		return fmt.Errorf("expected error return, got: %v", ft)
	}
	c.funcs[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	return nil
}

/**
 * SetStructFieldCopy - Declares an explicit name-mapping between fields in different structs.
 */
func (c *Converter) SetStructFieldCopy(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	st := reflect.TypeOf(srcFieldType)
	dt := reflect.TypeOf(destFieldType)
	srcKey := typeNamePair{st, srcFieldName}
	destKey := typeNamePair{dt, destFieldName}
	c.structFieldDests[srcKey] = append(c.structFieldDests[srcKey], destKey)
	c.structFieldSources[destKey] = append(c.structFieldSources[destKey], srcKey)
	return nil
}

type FieldMatchingFlags int

const (
	// DestFromSource - Default: Logic drives from destination fields to find sources.
	DestFromSource FieldMatchingFlags = 0
	// SourceToDest - Alternate: Logic drives from source fields to push to destinations.
	SourceToDest FieldMatchingFlags = 1 << iota
	IgnoreMissingFields
	AllowDifferentFieldTypeNames
)

func (f FieldMatchingFlags) IsSet(flag FieldMatchingFlags) bool {
	if flag == DestFromSource {
		return f&SourceToDest != SourceToDest
	}
	return f&flag == flag
}

/**
 * Convert - Entry point for data structure translation.
 * 
 * Logic: Validates writeability of the destination and establishes the 
 * recursive path-tracking context.
 */
func (c *Converter) Convert(src, dest interface{}, flags FieldMatchingFlags, meta *Meta) error {
	dv, err := EnforcePtr(dest)
	if err != nil {
		return err
	}
	if !dv.CanAddr() {
		return fmt.Errorf("can't write to dest")
	}
	sv, err := EnforcePtr(src)
	if err != nil {
		return err
	}
	s := &scope{
		converter: c,
		flags:     flags,
		meta:      meta,
	}
	// Initializing stacks to prevent underflow on root-level metadata access.
	s.srcStack.push(scopeStackElem{})
	s.destStack.push(scopeStackElem{})
	return c.convert(sv, dv, s)
}

/**
 * convert - Internal engine for deep recursive object copying.
 * 
 * Algorithm: Type-driven dispatch with expert-first precedence.
 * 1. Checks for registered expert handlers.
 * 2. Validates type name compatibility.
 * 3. Handles direct assignment/cast for primitives.
 * 4. Recursively process collections (Slices, Maps, Structs, Pointers).
 */
func (c *Converter) convert(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()
	
	// Block Logic: Expert function dispatch.
	if fv, ok := c.funcs[typePair{st, dt}]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Calling custom conversion of '%v' to '%v'", st, dt)
		}
		args := []reflect.Value{sv.Addr(), dv.Addr(), reflect.ValueOf(scope)}
		ret := fv.Call(args)[0].Interface()
		if ret == nil {
			return nil
		}
		return ret.(error)
	}

	// Logic: Safety check for type name parity unless explicitly relaxed.
	if !scope.flags.IsSet(AllowDifferentFieldTypeNames) && c.NameFunc(dt) != c.NameFunc(st) {
		return scope.error("type names don't match (%v, %v)", c.NameFunc(st), c.NameFunc(dt))
	}

	// Block Logic: Primitive assignment fast-path.
	if st.AssignableTo(dt) {
		dv.Set(sv)
		return nil
	}
	if st.ConvertibleTo(dt) {
		dv.Set(sv.Convert(dt))
		return nil
	}

	if c.Debug != nil {
		c.Debug.Logf("Trying to convert '%v' to '%v'", st, dt)
	}

	// Synchronization: Push current values onto path stacks before recursing.
	scope.srcStack.push(scopeStackElem{value: sv})
	scope.destStack.push(scopeStackElem{value: dv})
	defer scope.srcStack.pop()
	defer scope.destStack.pop()

	/**
	 * Block Logic: Container recursion.
	 * Invariant: Successfully populates 'dv' via element-wise translation of 'sv'.
	 */
	switch dv.Kind() {
	case reflect.Struct:
		return c.convertStruct(sv, dv, scope)
	case reflect.Slice:
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
		for i := 0; i < sv.Len(); i++ {
			scope.setIndices(i, i)
			if err := c.convert(sv.Index(i), dv.Index(i), scope); err != nil {
				return err
			}
		}
	case reflect.Ptr:
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.New(dt.Elem()))
		return c.convert(sv.Elem(), dv.Elem(), scope)
	case reflect.Map:
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeMap(dt))
		for _, sk := range sv.MapKeys() {
			dk := reflect.New(dt.Key()).Elem()
			if err := c.convert(sk, dk, scope); err != nil {
				return err
			}
			dkv := reflect.New(dt.Elem()).Elem()
			scope.setKeys(sk.Interface(), dk.Interface())
			if err := c.convert(sv.MapIndex(sk), dkv, scope); err != nil {
				return err
			}
			dv.SetMapIndex(dk, dkv)
		}
	default:
		return scope.error("couldn't copy '%v' into '%v'; unhandled type kind", st, dt)
	}
	return nil
}

/**
 * convertStruct - Logic for field-to-field struct mapping.
 */
func (c *Converter) convertStruct(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	listType := dt
	if scope.flags.IsSet(SourceToDest) {
		listType = st
	}
	
	// Block Logic: Field iteration.
	for i := 0; i < listType.NumField(); i++ {
		f := listType.Field(i)
		
		// Logic: Check for explicit mapping overrides first.
		if found, err := c.checkStructField(f.Name, sv, dv, scope); found {
			if err != nil {
				return err
			}
			continue
		}
		
		df := dv.FieldByName(f.Name)
		sf := sv.FieldByName(f.Name)
		
		// Metadata Propagation: Storing tags for use within expert handlers.
		if sf.IsValid() {
			field, _ := st.FieldByName(f.Name)
			scope.srcStack.top().tag = field.Tag
		}
		if df.IsValid() {
			field, _ := dt.FieldByName(f.Name)
			scope.destStack.top().tag = field.Tag
		}
		
		if !df.IsValid() || !sf.IsValid() {
			switch {
			case scope.flags.IsSet(IgnoreMissingFields):
				// No error.
			case scope.flags.IsSet(SourceToDest):
				return scope.error("%v not present in dest (%v to %v)", f.Name, st, dt)
			default:
				return scope.error("%v not present in src (%v to %v)", f.Name, st, dt)
			}
			continue
		}
		
		scope.srcStack.top().key = f.Name
		scope.destStack.top().key = f.Name
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

/**
 * checkStructField - Evaluates explicit field transformation rules.
 * 
 * Logic: Matches field names and types against registered correspondence rules 
 * to handle schema variations.
 */
func (c *Converter) checkStructField(fieldName string, sv, dv reflect.Value, scope *scope) (bool, error) {
	replacementMade := false
	
	// Block Logic: Destination-driven mapping.
	if scope.flags.IsSet(DestFromSource) {
		df := dv.FieldByName(fieldName)
		if !df.IsValid() {
			return false, nil
		}
		destKey := typeNamePair{df.Type(), fieldName}
		for _, potentialSourceKey := range c.structFieldSources[destKey] {
			sf := sv.FieldByName(potentialSourceKey.fieldName)
			if !sf.IsValid() {
				continue
			}
			if sf.Type() == potentialSourceKey.fieldType {
				scope.srcStack.top().key = potentialSourceKey.fieldName
				scope.destStack.top().key = fieldName
				if err := c.convert(sf, df, scope); err != nil {
					return true, err
				}
				replacementMade = true
			}
		}
		return replacementMade, nil
	}

	// Block Logic: Source-driven mapping.
	sf := sv.FieldByName(fieldName)
	if !sf.IsValid() {
		return false, nil
	}
	srcKey := typeNamePair{sf.Type(), fieldName}
	for _, potentialDestKey := range c.structFieldDests[srcKey] {
		df := dv.FieldByName(potentialDestKey.fieldName)
		if !df.IsValid() {
			continue
		}
		if df.Type() == potentialDestKey.fieldType {
			scope.srcStack.top().key = fieldName
			scope.destStack.top().key = potentialDestKey.fieldName
			if err := c.convert(sf, df, scope); err != nil {
				return true, err
			}
			replacementMade = true
		}
	}
	return replacementMade, nil
}
