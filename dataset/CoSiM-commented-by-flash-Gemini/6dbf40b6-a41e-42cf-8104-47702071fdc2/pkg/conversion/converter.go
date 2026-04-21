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
 * @brief Generic reflection-based type conversion engine for Kubernetes.
 * 
 * Functional Intent: Provides a flexible framework for translating data structures 
 * between different API versions or internal representations. It handles deep 
 * copying of nested objects, supports custom conversion functions for specific 
 * type pairs, and allows for explicit field-to-field mapping between structs 
 * with different schemas.
 * 
 * Domain: Production Systems, Data Serialization, Version Migration.
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
 * @brief Hook for high-verbosity diagnostic logging during complex conversion paths.
 */
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

/**
 * @struct Converter
 * @brief State container for registered conversion functions and field mapping rules.
 * 
 * Logic: Orchestrates the transformation process by dispatching to specialized 
 * functions or falling back to a reflection-driven recursive copy algorithm.
 */
type Converter struct {
	// funcs - Registry of expert conversion logic for specific (SrcType, DestType) pairs.
	funcs map[typePair]reflect.Value

	// structFieldDests - Forward mapping rules (SourceField -> DestinationField).
	structFieldDests map[typeNamePair][]typeNamePair

	// structFieldSources - Reverse mapping rules (DestinationField -> SourceField).
	structFieldSources map[typeNamePair][]typeNamePair

	Debug DebugLogger

	// NameFunc - Customizable discriminator for type identity during matching checks.
	NameFunc func(t reflect.Type) string
}

/**
 * NewConverter - Factory for initializing a fresh conversion context.
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
 * @brief Control handle passed to conversion functions to allow recursive sub-object processing.
 * 
 * Functional Utility: Decouples the expert conversion logic from the global 
 * converter state, enabling modularity and stack-safe recursion.
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
 * @brief Contextual metadata (e.g. version identifiers) passed through the conversion pipeline.
 */
type Meta struct {
	SrcVersion  string
	DestVersion string
}

/**
 * @struct scope
 * @brief Internal implementation of the Scope interface, tracking stack depth for debugging.
 */
type scope struct {
	converter *Converter
	meta      *Meta
	flags     FieldMatchingFlags

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
 * describe - Computes a human-readable path string for the current recursive depth.
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

func (s *scope) setIndices(src, dest int) {
	s.srcStack.top().key = fmt.Sprintf("[%v]", src)
	s.destStack.top().key = fmt.Sprintf("[%v]", dest)
}

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

func (s *scope) error(message string, args ...interface{}) error {
	srcPath, destPath := s.describe()
	where := fmt.Sprintf("converting %v to %v: ", srcPath, destPath)
	return fmt.Errorf(where+message, args...)
}

/**
 * Register - Adds a specialized conversion handler to the engine.
 * 
 * Logic: Validates the function signature to ensure it matches the 
 * (InPtr, OutPtr, Scope) -> Error pattern required for dynamic dispatch.
 */
func (c *Converter) Register(conversionFunc interface{}) error {
	fv := reflect.ValueOf(conversionFunc)
	ft := fv.Type()
	
	// Pre-condition: Input must be a function with specific arity.
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
	
	// Invariant: Stores the handler for O(1) lookup during conversion dispatch.
	c.funcs[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	return nil
}

/**
 * SetStructFieldCopy - Registers an explicit field-name mapping between structs.
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
	// DestFromSource - Default logic: Iterate destination and find matching sources.
	DestFromSource FieldMatchingFlags = 0
	// SourceToDest - Mapping logic: Iterate source and push into matching destinations.
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
 * Logic: Validates input pointers and initializes the recursive stack. 
 * Not thread-safe for objects with circular references.
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
	s.srcStack.push(scopeStackElem{})
	s.destStack.push(scopeStackElem{})
	return c.convert(sv, dv, s)
}

/**
 * convert - Internal recursive engine for deep object copying.
 * 
 * Algorithm: Type-driven dispatch.
 * 1. Checks for expert conversion functions.
 * 2. Validates type compatibility.
 * 3. Handles primitives (assign/convert).
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

	// Logic: Strict type identity check unless overridden by flags.
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

	scope.srcStack.push(scopeStackElem{value: sv})
	scope.destStack.push(scopeStackElem{value: dv})
	defer scope.srcStack.pop()
	defer scope.destStack.pop()

	/**
	 * Block Logic: Container-type recursion.
	 * Invariant: Successfully populates 'dv' by performing element-wise 
	 * conversion from 'sv'.
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
		return scope.error("couldn't copy '%v' into '%v'; didn't understand types", st, dt)
	}
	return nil
}

/**
 * convertStruct - Iterates and converts individual fields of a struct.
 */
func (c *Converter) convertStruct(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	listType := dt
	if scope.flags.IsSet(SourceToDest) {
		listType = st
	}
	
	/**
	 * Block Logic: Field iteration loop.
	 * Invariant: Every field in the target struct is populated either from 
	 * a matching field in the source or via an explicit mapping rule.
	 */
	for i := 0; i < listType.NumField(); i++ {
		f := listType.Field(i)
		
		// Logic: Check for explicit field-level overrides before falling back to name-matching.
		if found, err := c.checkStructField(f.Name, sv, dv, scope); found {
			if err != nil {
				return err
			}
			continue
		}
		
		df := dv.FieldByName(f.Name)
		sf := sv.FieldByName(f.Name)
		
		// Synchronization: Metadata (StructTags) propagation.
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
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

/**
 * checkStructField - Resolves explicit mapping rules for a named field.
 */
func (c *Converter) checkStructField(fieldName string, sv, dv reflect.Value, scope *scope) (bool, error) {
	replacementMade := false
	
	// Block Logic: Destination-driven rule evaluation.
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

	// Block Logic: Source-driven rule evaluation.
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
