/**
 * @file converter.go
 * @brief Provides a generic, reflection-based object conversion library.
 *
 * @details This file defines a powerful and flexible conversion system that uses Go's reflection
 * capabilities to copy data between objects of different types. It is particularly useful in
 * systems that require API versioning (like Kubernetes), where it's necessary to convert
 * between internal and versioned representations of the same logical data structure.
 *
 * Key Features:
 * - Custom Conversion Functions: Users can register their own functions to handle complex
 *   or non-standard conversions between specific types.
 * - Automatic Struct Conversion: By default, it can copy fields between structs that have
 *   the same name and compatible types.
 * - Field Mapping: Provides a mechanism (`SetStructFieldCopy`) to explicitly map a field
 *   in a source struct to a field with a different name or type in a destination struct.
 * - Fine-grained Control: Conversion behavior can be modified using `FieldMatchingFlags`
 *   to control aspects like field matching direction and error handling for missing fields.
 * - Scope-aware Recursion: Custom conversion functions can recursively call the converter
 *   to handle nested objects, while maintaining access to the overall conversion context (like
 *   struct tags and metadata).
 *
 * Algorithm:
 * The conversion process is recursive. It starts with the top-level `Convert` call. The private `convert`
 * method then inspects the types of the source and destination objects.
 * 1. If a custom conversion function is registered for the `(source type, dest type)` pair, it is invoked.
 * 2. If not, the system attempts a direct assignment or type conversion if the types are compatible.
 * 3. If the types are complex (structs, slices, maps, pointers), it recursively applies the conversion
 *    logic to their constituent elements. For structs, it iterates through fields and attempts to match
 *    them based on names or explicit mapping rules.
 * This process continues until all parts of the object graph have been visited and converted.
 */

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

package conversion

import (
	"fmt"
	"reflect"
)

// typePair is a key for maps, representing a source and destination type.
type typePair struct {
	source reflect.Type
	dest   reflect.Type
}

// typeNamePair is a key for maps, representing a field's type and name.
type typeNamePair struct {
	fieldType reflect.Type
	fieldName string
}

// DebugLogger allows you to get debugging messages if necessary.
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

// Converter knows how to convert one type to another. It holds the registry of
// custom conversion functions and field mappings.
type Converter struct {
	// Map from a conversion pair to a function which can do the conversion.
	funcs map[typePair]reflect.Value

	// This is a map from a source field type and name to a list of destination
	// field types and names. It's used for explicit field mapping.
	structFieldDests map[typeNamePair][]typeNamePair

	// This is the reverse of `structFieldDests`, allowing lookups from destination to source.
	// This makes `SourceToDest` flag work for field remapping.
	structFieldSources map[typeNamePair][]typeNamePair

	// If non-nil, will be called to print helpful debugging info. Quite verbose.
	Debug DebugLogger

	// NameFunc is called to retrieve a user-defined name of a type.
	// This name is used to decide if two types can be converted by default.
	// The default implementation returns the Go type name.
	NameFunc func(t reflect.Type) string
}

// NewConverter creates a new, initialized Converter object.
func NewConverter() *Converter {
	return &Converter{
		funcs:              map[typePair]reflect.Value{},
		NameFunc:           func(t reflect.Type) string { return t.Name() },
		structFieldDests:   map[typeNamePair][]typeNamePair{},
		structFieldSources: map[typeNamePair][]typeNamePair{},
	}
}

// Scope is passed to conversion funcs to allow them to continue an ongoing conversion.
// It provides access to the parent converter, metadata, and context about the
// current position in the object graph (e.g., struct tags).
type Scope interface {
	// Call Convert to recursively convert sub-objects.
	Convert(src, dest interface{}, flags FieldMatchingFlags) error

	// SrcTag and DestTag return the struct tags of the fields being converted.
	SrcTag() reflect.StructTag
	DestTag() reflect.StructTag

	// Flags returns the flags with which the conversion was started.
	Flags() FieldMatchingFlags

	// Meta returns any high-level information originally passed to Convert.
	Meta() *Meta
}

// Meta is supplied by a higher-level framework (like a Scheme) when it calls Convert.
// It can be used to pass contextual information, like API versions, to conversion funcs.
type Meta struct {
	SrcVersion  string
	DestVersion string
}

// scope contains internal information about an ongoing conversion.
type scope struct {
	converter    *Converter
	meta         *Meta
	flags        FieldMatchingFlags
	srcTagStack  []reflect.StructTag
	destTagStack []reflect.StructTag
}

// push adds a level to the src/dest tag stacks, for tracking nested conversions.
func (s *scope) push() {
	s.srcTagStack = append(s.srcTagStack, "")
	s.destTagStack = append(s.destTagStack, "")
}

// pop removes a level from the src/dest tag stacks.
func (s *scope) pop() {
	n := len(s.srcTagStack)
	s.srcTagStack = s.srcTagStack[:n-1]
	s.destTagStack = s.destTagStack[:n-1]
}

func (s *scope) setSrcTag(tag reflect.StructTag) {
	s.srcTagStack[len(s.srcTagStack)-1] = tag
}

func (s.scope) setDestTag(tag reflect.StructTag) {
	s.destTagStack[len(s.destTagStack)-1] = tag
}

// Convert continues a conversion recursively.
func (s *scope) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.Convert(src, dest, flags, s.meta)
}

// SrcTag returns the tag of the struct containing the current source item, if any.
func (s *scope) SrcTag() reflect.StructTag {
	return s.srcTagStack[len(s.srcTagStack)-1]
}

// DestTag returns the tag of the struct containing the current dest item, if any.
func (s *scope) DestTag() reflect.StructTag {
	return s.destTagStack[len(s.destTagStack)-1]
}

// Flags returns the flags with which the current conversion was started.
func (s *scope) Flags() FieldMatchingFlags {
	return s.flags
}

// Meta returns the meta object that was originally passed to Convert.
func (s *scope) Meta() *Meta {
	return s.meta
}

// Register registers a custom conversion func with the Converter.
// The conversionFunc must have a signature of `func(in *T, out *U, s Scope) error`,
// where T and U are the source and destination types.
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

// SetStructFieldCopy registers a mapping between a source field and a destination field.
// This is used when fields are named differently across types but should be copied.
func (c *Converter) SetStructFieldCopy(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	st := reflect.TypeOf(srcFieldType)
	dt := reflect.TypeOf(destFieldType)
	srcKey := typeNamePair{st, srcFieldName}
	destKey := typeNamePair{dt, destFieldName}
	c.structFieldDests[srcKey] = append(c.structFieldDests[srcKey], destKey)
	c.structFieldSources[destKey] = append(c.structFieldSources[destKey], srcKey)
	return nil
}

// FieldMatchingFlags contains a list of options for how struct fields are copied.
type FieldMatchingFlags int

const (
	// DestFromSource is the default behavior. It loops through destination fields
	// and looks for a matching field in the source to copy from.
	DestFromSource FieldMatchingFlags = 0
	// SourceToDest loops through source fields and looks for a matching field
	// in the destination to copy to.
	SourceToDest FieldMatchingFlags = 1 << iota
	// IgnoreMissingFields prevents an error from being returned if a matching
	// field cannot be found in the source or destination.
	IgnoreMissingFields
	// AllowDifferentFieldTypeNames allows conversion between fields even if their
	// underlying types have different names (but are otherwise convertible).
	AllowDifferentFieldTypeNames
)

// IsSet returns true if the given flag or combination of flags is set.
func (f FieldMatchingFlags) IsSet(flag FieldMatchingFlags) bool {
	if flag == DestFromSource {
		// The default value has no bit set, so handle it separately.
		return f&SourceToDest != SourceToDest
	}
	return f&flag == flag
}

// Convert will translate src to dest if it knows how. Both must be pointers.
// It initiates the recursive conversion process. Not safe for objects with cyclic references!
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
	s.push() // Easy way to make SrcTag and DestTag never fail
	return c.convert(sv, dv, s)
}

// convert is the core recursive conversion function.
func (c *Converter) convert(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	// Block Logic: Check for and execute a custom conversion function if one is registered.
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

	// Pre-condition: Unless explicitly allowed, types must have matching names for default conversion.
	if !scope.flags.IsSet(AllowDifferentFieldTypeNames) && c.NameFunc(dt) != c.NameFunc(st) {
		return fmt.Errorf("can't convert %v to %v because type names don't match (%v, %v).", st, dt, c.NameFunc(st), c.NameFunc(dt))
	}

	// Block Logic: Handle simple types that are directly assignable or convertible.
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

	scope.push()
	defer scope.pop()

	// Block Logic: Dispatch to type-specific conversion logic.
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

// convertStruct handles the automatic conversion of struct types.
func (c *Converter) convertStruct(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	// Determine which struct's fields to iterate over, based on flags.
	listType := dt
	if scope.flags.IsSet(SourceToDest) {
		listType = st
	}
	// Invariant: Loop iterates through all fields of the chosen `listType`.
	for i := 0; i < listType.NumField(); i++ {
		f := listType.Field(i)
		// Block Logic: First, check if a custom field mapping rule applies to this field.
		if found, err := c.checkStructField(f.Name, sv, dv, scope); found {
			if err != nil {
				return err
			}
			continue // If a custom rule was applied, skip default processing.
		}

		// Block Logic: Fallback to default behavior: copy fields with matching names.
		df := dv.FieldByName(f.Name)
		sf := sv.FieldByName(f.Name)
		if sf.IsValid() {
			field, _ := st.FieldByName(f.Name)
			scope.setSrcTag(field.Tag)
		}
		if df.IsValid() {
			field, _ := dt.FieldByName(f.Name)
			scope.setDestTag(field.Tag)
		}

		// Pre-condition: Check if the field exists in both source and destination.
		if !df.IsValid() || !sf.IsValid() {
			if scope.flags.IsSet(IgnoreMissingFields) == false {
				// Return an error if missing fields are not ignored.
				if scope.flags.IsSet(SourceToDest) {
					return fmt.Errorf("%v not present in dest (%v to %v)", f.Name, st, dt)
				}
				return fmt.Errorf("%v not present in src (%v to %v)", f.Name, st, dt)
			}
			continue
		}
		// Recursively convert the matched field.
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

// checkStructField returns true if a custom field mapping rule was found and applied.
// The error should be ignored if it returns false.
func (c *Converter) checkStructField(fieldName string, sv, dv reflect.Value, scope *scope) (bool, error) {
	replacementMade := false
	if scope.flags.IsSet(DestFromSource) {
		df := dv.FieldByName(fieldName)
		if !df.IsValid() {
			return false, nil
		}
		destKey := typeNamePair{df.Type(), fieldName}
		// Check each of the potential source (type, name) pairs to see if they're present in sv.
		for _, potentialSourceKey := range c.structFieldSources[destKey] {
			sf := sv.FieldByName(potentialSourceKey.fieldName)
			if sf.IsValid() && sf.Type() == potentialSourceKey.fieldType {
				// Both the source's name and type matched, so copy.
				if err := c.convert(sf, df, scope); err != nil {
					return true, err
				}
				replacementMade = true
			}
		}
		return replacementMade, nil
	}

	// This branch handles SourceToDest mapping.
	sf := sv.FieldByName(fieldName)
	if !sf.IsValid() {
		return false, nil
	}
	srcKey := typeNamePair{sf.Type(), fieldName}
	// Check each of the potential dest (type, name) pairs to see if they're present in dv.
	for _, potentialDestKey := range c.structFieldDests[srcKey] {
		df := dv.FieldByName(potentialDestKey.fieldName)
		if df.IsValid() && df.Type() == potentialDestKey.fieldType {
			// Both the dest's name and type matched, so copy.
			if err := c.convert(sf, df, scope); err != nil {
				return true, err
			}
			replacementMade = true
		}
	}
	return replacementMade, nil
}
