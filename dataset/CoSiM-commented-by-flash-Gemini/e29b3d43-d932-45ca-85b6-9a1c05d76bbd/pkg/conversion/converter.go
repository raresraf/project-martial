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
 * @brief Generic reflection-based object translation engine for schema migration.
 * 
 * Functional Intent: Facilitates deep transformation between different versions 
 * of data structures (e.g. Kubernetes API objects). It supports both automatic 
 * structural reflection and specialized expert handlers for complex type pairs. 
 * The system maintains dual stacks for source and destination struct tags, 
 * ensuring that contextual metadata is available to custom conversion functions 
 * at any depth in the object hierarchy.
 * 
 * Domain: Production Systems, Schema Evolution, Data Mapping.
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
 * @brief Diagnostic hook for tracing recursive conversion logic paths.
 */
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

/**
 * @struct Converter
 * @brief Central authority for registered conversion functions and field mapping rules.
 * 
 * Logic: Dispatches transformation requests to expert functions or falls back 
 * to a recursive, structural reflection engine.
 */
type Converter struct {
	// funcs - Registry of hand-optimized conversion logic for specific type pairs.
	funcs map[typePair]reflect.Value

	// structFieldDests - Explicit mapping rules for forward transformation (Src -> Dest).
	structFieldDests map[typeNamePair][]typeNamePair

	// structFieldSources - Explicit mapping rules for backward transformation (Dest <- Src).
	structFieldSources map[typeNamePair][]typeNamePair

	Debug DebugLogger

	// NameFunc - Customizable logic for determining type identity compatibility.
	NameFunc func(t reflect.Type) string
}

/**
 * NewConverter - Factory for initializing a fresh converter with default type naming.
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
 * @brief Thread-safe handle providing recursion control and metadata access to conversion functions.
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
 * @brief State container for high-level conversion context (e.g. version identifiers).
 */
type Meta struct {
	SrcVersion  string
	DestVersion string
}

/**
 * @struct scope
 * @brief Internal implementation of Scope, managing synchronized tag stacks for deep objects.
 */
type scope struct {
	converter *Converter
	meta      *Meta
	flags     FieldMatchingFlags

	// srcStack & destStack - Dual stacks to preserve structural context during recursion.
	srcStack  scopeStack
	destStack scopeStack
}

type scopeStackElem struct {
	tag   reflect.StructTag
	value reflect.Value
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

/**
 * Register - Adds a specialized transformation handler to the converter.
 * 
 * Logic: Validates input/output types to ensure compliance with the 
 * (InPtr, OutPtr, Scope) -> error dispatch pattern.
 */
func (c *Converter) Register(conversionFunc interface{}) error {
	fv := reflect.ValueOf(conversionFunc)
	ft := fv.Type()
	
	// Pre-condition: Input must be a function with specific arity and pointer args.
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
	
	// Invariant: Stores the handler for O(1) retrieval during conversion.
	c.funcs[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	return nil
}

/**
 * SetStructFieldCopy - Declares an explicit correspondence between differently-named struct fields.
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
	// DestFromSource - Default logic: Iterate destination and look for sources.
	DestFromSource FieldMatchingFlags = 0
	// SourceToDest - Mapping logic: Iterate source and push to destinations.
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
 * Convert - Entry point for object translation.
 * 
 * Logic: Validates pointers and initializes the recursive stack context. 
 * Note: Not safe for objects with circular references.
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
 * convert - Internal engine for recursive structural copying.
 * 
 * Algorithm: Type-driven dispatch.
 * 1. Checks for expert handlers.
 * 2. Validates type compatibility.
 * 3. Handles primitives via direct assignment or cast.
 * 4. Recursively processes containers (Structs, Slices, Maps, Pointers).
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

	// Logic: Safety check for type name parity.
	if !scope.flags.IsSet(AllowDifferentFieldTypeNames) && c.NameFunc(dt) != c.NameFunc(st) {
		return fmt.Errorf("can't convert %v to %v because type names don't match (%v, %v).", st, dt, c.NameFunc(st), c.NameFunc(dt))
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

	// Synchronization: Snapshotting state onto stacks before recursing.
	scope.srcStack.push(scopeStackElem{value: sv})
	scope.destStack.push(scopeStackElem{value: dv})
	defer scope.srcStack.pop()
	defer scope.destStack.pop()

	/**
	 * Block Logic: Container type recursion.
	 * Invariant: Successfully populates 'dv' by performing element-wise 
	 * translation of 'sv'.
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
			if err := c.convert(sv.MapIndex(sk), dkv, scope); err != nil {
				return err
			}
			dv.SetMapIndex(dk, dkv)
		}
	default:
		return fmt.Errorf("couldn't copy '%v' into '%v'", st, dt)
	}
	return nil
}

/**
 * convertStruct - Iterates and converts individual struct fields.
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
		
		// Logic: Check for explicit field name mapping rules before name-matching.
		if found, err := c.checkStructField(f.Name, sv, dv, scope); found {
			if err != nil {
				return err
			}
			continue
		}
		
		df := dv.FieldByName(f.Name)
		sf := sv.FieldByName(f.Name)
		
		// Metadata Propagation: Capturing struct tags to the stack tops.
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
				return fmt.Errorf("%v not present in dest (%v to %v)", f.Name, st, dt)
			default:
				return fmt.Errorf("%v not present in src (%v to %v)", f.Name, st, dt)
			}
			continue
		}
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

/**
 * checkStructField - Evaluates explicit field-level correspondence rules.
 */
func (c *Converter) checkStructField(fieldName string, sv, dv reflect.Value, scope *scope) (bool, error) {
	replacementMade := false
	
	// Block Logic: Destination-driven lookup.
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
				if err := c.convert(sf, df, scope); err != nil {
					return true, err
				}
				replacementMade = true
			}
		}
		return replacementMade, nil
	}

	// Block Logic: Source-driven lookup.
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
			if err := c.convert(sf, df, scope); err != nil {
				return true, err
			}
			replacementMade = true
		}
	}
	return replacementMade, nil
}
