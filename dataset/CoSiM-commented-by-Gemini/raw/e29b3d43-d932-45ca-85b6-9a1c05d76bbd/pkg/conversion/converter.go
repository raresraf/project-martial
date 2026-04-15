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
 * @brief Implements a reflection-based deep object conversion library.
 *
 * This package provides a generic `Converter` capable of recursively copying and transforming
 * data between Go types. It supports custom conversion functions, field name mapping, and
 * various matching strategies, making it suitable for tasks like API versioning where data
 * structures evolve over time. The core logic uses reflection to traverse object graphs and
 * apply conversion rules dynamically.
 */


package conversion

import (
	"fmt"
	"reflect"
)

// typePair holds a source and destination type, and is used as a key for lookup
// in the custom conversion function map.
type typePair struct {
	source reflect.Type
	dest   reflect.Type
}

// typeNamePair holds a type and a name, and is used as a key for overriding
// struct field-to-field copying.
type typeNamePair struct {
	fieldType reflect.Type
	fieldName string
}

// DebugLogger allows you to get debugging messages if necessary.
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

// Converter knows how to convert one type to another.
type Converter struct {
	// funcs is a map from a conversion pair to a function which can
	// do the conversion.
	funcs map[typePair]reflect.Value

	// structFieldDests is a map from a source field type and name, to a list of destination
	// field type and names. It is used to implement custom field mappings.
	structFieldDests map[typeNamePair][]typeNamePair

	// structFieldSources provides a reverse lookup for structFieldDests, enabling
	// conversions to be specified from destination to source as well.
	structFieldSources map[typeNamePair][]typeNamePair

	// If non-nil, Debug will be called to print helpful debugging info. The output is verbose.
	Debug DebugLogger

	// NameFunc is called to retrieve the name of a type. This name is used for
	// deciding whether two types match and a conversion should be attempted.
	// The default implementation returns the Go type name.
	NameFunc func(t reflect.Type) string
}

// NewConverter creates a new Converter object with default settings.
func NewConverter() *Converter {
	return &Converter{
		funcs:              map[typePair]reflect.Value{},
		NameFunc:           func(t reflect.Type) string { return t.Name() },
		structFieldDests:   map[typeNamePair][]typeNamePair{},
		structFieldSources: map[typeNamePair][]typeNamePair{},
	}
}

// Scope is passed to conversion funcs to allow them to continue an ongoing conversion.
// If multiple converters exist in the system, Scope will allow you to use the correct one
// from a conversion function--that is, the one your conversion function was called by.
type Scope interface {
	// Convert calls the converter to translate sub-objects. Calling it with the same
	// parameters as the enclosing function will lead to infinite recursion.
	Convert(src, dest interface{}, flags FieldMatchingFlags) error

	// SrcTag and DestTag return the struct tags of the source and destination fields, respectively.
	// These are empty if the enclosing object is not a struct.
	SrcTag() reflect.StructTag
	DestTag() reflect.StructTag

	// Flags returns the flags with which the conversion was started.
	Flags() FieldMatchingFlags

	// Meta returns any context-specific information originally passed to Convert.
	Meta() *Meta
}

// Meta is supplied by a higher-level framework (like a Scheme) when it calls Convert.
// It carries metadata about the conversion, such as source and destination API versions.
type Meta struct {
	SrcVersion  string
	DestVersion string

	// TODO: If needed, add a user data field here for extensibility.
}

// scope contains information about an ongoing conversion, acting as the internal
// implementation of the Scope interface.
type scope struct {
	converter *Converter
	meta      *Meta
	flags     FieldMatchingFlags

	// srcStack & destStack track the traversal path through the source and destination
	// object graphs. They are separate because the structures may not be identical.
	srcStack  scopeStack
	destStack scopeStack
}

// scopeStackElem represents a single level in the object graph traversal,
// holding the value and struct tag of the current field.
type scopeStackElem struct {
	tag   reflect.StructTag
	value reflect.Value
}

// scopeStack is a stack data structure for tracking the conversion path.
type scopeStack []scopeStackElem

// pop removes the top element from the stack.
func (s *scopeStack) pop() {
	n := len(*s)
	*s = (*s)[:n-1]
}

// push adds an element to the top of the stack.
func (s *scopeStack) push(e scopeStackElem) {
	*s = append(*s, e)
}

// top returns the top element of the stack without removing it.
func (s *scopeStack) top() *scopeStackElem {
	return &(*s)[len(*s)-1]
}

// Convert continues a conversion on a sub-object using the parent's scope.
func (s *scope) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.Convert(src, dest, flags, s.meta)
}

// SrcTag returns the tag of the struct containing the current source item, if any.
func (s *scope) SrcTag() reflect.StructTag {
	return s.srcStack.top().tag
}

// DestTag returns the tag of the struct containing the current dest item, if any.
func (s *scope) DestTag() reflect.StructTag {
	return s.destStack.top().tag
}

// Flags returns the flags with which the current conversion was started.
func (s *scope) Flags() FieldMatchingFlags {
	return s.flags
}

// Meta returns the meta object that was originally passed to Convert.
func (s *scope) Meta() *Meta {
	return s.meta
}

// Register registers a conversion func with the Converter. conversionFunc must take
// three parameters: a pointer to the input type, a pointer to the output type, and
// a conversion.Scope (which should be used if recursive conversion calls are desired).
// It must return an error.
//
// Example:
// c.Register(func(in *Pod, out *v1beta1.Pod, s Scope) error { ... return nil })
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
	// This convolution is necessary, otherwise TypeOf picks up on the fact
	// that forErrorType is nil.
	errorType := reflect.TypeOf(&forErrorType).Elem()
	if ft.Out(0) != errorType {
		return fmt.Errorf("expected error return, got: %v", ft)
	}
	c.funcs[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	return nil
}

// SetStructFieldCopy registers a correspondence between struct fields. When a field
// with a type and name matching srcFieldType and srcFieldName is encountered, its value
// will be copied to the destination field matching destFieldType and destFieldName.
// This allows for field renaming and restructuring between types.
// It can be called multiple times for the same source field to broadcast its value.
func (c *Converter) SetStructFieldCopy(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	st := reflect.TypeOf(srcFieldType)
	dt := reflect.TypeOf(destFieldType)
	srcKey := typeNamePair{st, srcFieldName}
	destKey := typeNamePair{dt, destFieldName}
	c.structFieldDests[srcKey] = append(c.structFieldDests[srcKey], destKey)
	c.structFieldSources[destKey] = append(c.structFieldSources[destKey], srcKey)
	return nil
}

// FieldMatchingFlags controls the behavior of struct field copying.
// These constants can be bitwise OR'd together.
type FieldMatchingFlags int

const (
	// DestFromSource (default) iterates through destination fields and looks for a
	// matching field in the source to copy from. Source fields without a
	// corresponding destination field are ignored.
	DestFromSource FieldMatchingFlags = 0
	// SourceToDest iterates through source fields and looks for a matching
	// field in the destination to copy to. Destination fields without a
	// corresponding source field are ignored. This flag overrides DestFromSource.
	SourceToDest FieldMatchingFlags = 1 << iota
	// IgnoreMissingFields prevents an error from being returned if a corresponding
	// source or destination field cannot be found.
	IgnoreMissingFields
	// AllowDifferentFieldTypeNames relaxes the requirement that field types must
	// have the same name for a conversion to be attempted.
	AllowDifferentFieldTypeNames
)

// IsSet returns true if the given flag or combination of flags is set in the bitmask.
func (f FieldMatchingFlags) IsSet(flag FieldMatchingFlags) bool {
	if flag == DestFromSource {
		// The bit logic doesn't work on the default value (0).
		return f&SourceToDest != SourceToDest
	}
	return f&flag == flag
}

// Convert translates src to dest if a conversion path is available.
// Both src and dest must be pointers. An error is returned if no conversion
// function is registered and the default reflection-based copying fails.
// 'meta' allows passing contextual information to conversion functions but is not
// used directly by Convert, other than being stored in the scope.
// This function is not safe for objects with cyclic references.
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
	// Push a dummy element onto the stack so that tag getters do not fail on the top-level object.
	s.srcStack.push(scopeStackElem{})
	s.destStack.push(scopeStackElem{})
	return c.convert(sv, dv, s)
}

// convert is the core recursive conversion function. It orchestrates the entire
// process by first checking for a custom conversion function. If none is found,
// it attempts a direct assignment or conversion for simple types. For complex types
// (structs, slices, pointers, maps), it recursively applies the conversion logic to
// their elements.
func (c *Converter) convert(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	// First, see if a custom conversion function is registered.
	if fv, ok := c.funcs[typePair{st, dt}]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Calling custom conversion of '%v' to '%v'", st, dt)
		}
		args := []reflect.Value{sv.Addr(), dv.Addr(), reflect.ValueOf(scope)}
		ret := fv.Call(args)[0].Interface()
		// This convolution is necessary because nil interfaces won't convert
		// to errors.
		if ret == nil {
			return nil
		}
		return ret.(error)
	}

	// If type names don't match, and we're not explicitly allowing it, fail.
	if !scope.flags.IsSet(AllowDifferentFieldTypeNames) && c.NameFunc(dt) != c.NameFunc(st) {
		return fmt.Errorf("can't convert %v to %v because type names don't match (%v, %v).", st, dt, c.NameFunc(st), c.NameFunc(dt))
	}

	// Handle primitive types and types that are directly assignable or convertible.
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

	// Push the current object onto the stack to track recursion depth and provide context.
	scope.srcStack.push(scopeStackElem{value: sv})
	scope.destStack.push(scopeStackElem{value: dv})
	defer scope.srcStack.pop()
	defer scope.destStack.pop()

	// Handle complex types.
	switch dv.Kind() {
	case reflect.Struct:
		return c.convertStruct(sv, dv, scope)
	case reflect.Slice:
		// Pre-condition: Source slice must not be nil to avoid creating an empty slice where nil is intended.
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt)) // Explicitly set to nil.
			return nil
		}
		// Invariant: A new slice is created with the same length and capacity as the source.
		dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
		for i := 0; i < sv.Len(); i++ {
			if err := c.convert(sv.Index(i), dv.Index(i), scope); err != nil {
				return err
			}
		}
	case reflect.Ptr:
		// Pre-condition: Source pointer must not be nil.
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt)) // Propagate nil pointer.
			return nil
		}
		// Invariant: A new pointer to a zero value of the destination element type is created.
		dv.Set(reflect.New(dt.Elem()))
		return c.convert(sv.Elem(), dv.Elem(), scope)
	case reflect.Map:
		// Pre-condition: Source map must not be nil.
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt)) // Propagate nil map.
			return nil
		}
		// Invariant: A new map is created, and each key-value pair is recursively converted.
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

// convertStruct handles the recursive conversion of struct types. It iterates through
// fields based on the specified FieldMatchingFlags (either source-to-dest or dest-to-source)
// and attempts to copy values, first checking for custom field mappings.
func (c *Converter) convertStruct(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	// Determine which struct's fields to iterate over based on the flags.
	listType := dt
	if scope.flags.IsSet(SourceToDest) {
		listType = st
	}
	for i := 0; i < listType.NumField(); i++ {
		f := listType.Field(i)

		// First, check if a custom field copy rule applies to this field.
		if found, err := c.checkStructField(f.Name, sv, dv, scope); found {
			if err != nil {
				return err
			}
			continue
		}

		// If no custom rule, fall back to same-name field copying.
		df := dv.FieldByName(f.Name)
		sf := sv.FieldByName(f.Name)
		if sf.IsValid() {
			field, _ := st.FieldByName(f.Name)
			scope.srcStack.top().tag = field.Tag
		}
		if df.IsValid() {
			field, _ := dt.FieldByName(f.Name)
			scope.destStack.top().tag = field.Tag
		}
		// Pre-condition: Source and destination fields must both be valid unless IgnoreMissingFields is set.
		if !df.IsValid() || !sf.IsValid() {
			if !scope.flags.IsSet(IgnoreMissingFields) {
				if scope.flags.IsSet(SourceToDest) {
					return fmt.Errorf("%v not present in dest (%v to %v)", f.Name, st, dt)
				}
				return fmt.Errorf("%v not present in src (%v to %v)", f.Name, st, dt)
			}
			continue
		}

		// Invariant: The field value is recursively converted.
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

// checkStructField checks if the given field matches a registered struct field copy rule.
// It returns true if a rule was found and applied, along with any error that occurred
// during the copy. If it returns false, the error should be ignored.
func (c *Converter) checkStructField(fieldName string, sv, dv reflect.Value, scope *scope) (bool, error) {
	replacementMade := false

	// Logic for when iterating destination fields (DestFromSource flag).
	if scope.flags.IsSet(DestFromSource) {
		df := dv.FieldByName(fieldName)
		if !df.IsValid() {
			return false, nil
		}
		destKey := typeNamePair{df.Type(), fieldName}
		// Check each potential source (type, name) pair to see if it's present in sv.
		for _, potentialSourceKey := range c.structFieldSources[destKey] {
			sf := sv.FieldByName(potentialSourceKey.fieldName)
			if !sf.IsValid() {
				continue
			}
			// Pre-condition: Source field type must match the registered rule.
			if sf.Type() == potentialSourceKey.fieldType {
				// Both source name and type matched, so perform the copy.
				if err := c.convert(sf, df, scope); err != nil {
					return true, err
				}
				replacementMade = true
			}
		}
		return replacementMade, nil
	}

	// Logic for when iterating source fields (SourceToDest flag).
	sf := sv.FieldByName(fieldName)
	if !sf.IsValid() {
		return false, nil
	}
	srcKey := typeNamePair{sf.Type(), fieldName}
	// Check each potential destination (type, name) pair to see if it's present in dv.
	for _, potentialDestKey := range c.structFieldDests[srcKey] {
		df := dv.FieldByName(potentialDestKey.fieldName)
		if !df.IsValid() {
			continue
		}
		// Pre-condition: Destination field type must match the registered rule.
		if df.Type() == potentialDestKey.fieldType {
			// Both destination name and type matched, so perform the copy.
			if err := c.convert(sf, df, scope); err != nil {
				return true, err
			}
			replacementMade = true
		}
	}
	return replacementMade, nil
}
