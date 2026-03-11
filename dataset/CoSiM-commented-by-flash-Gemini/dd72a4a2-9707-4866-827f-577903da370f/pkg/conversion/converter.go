// package conversion provides a generic type conversion utility, primarily using reflection.
// It is designed to convert data structures between different Go types, often used
// for API versioning or object model transformations.

package conversion

import (
	"fmt"
	"reflect"
)

// typePair represents a pair of source and destination reflection.Type values,
// used as a key in a map to store conversion functions.
type typePair struct {
	source reflect.Type
	dest   reflect.Type
}

// typeNamePair represents a combination of a field's reflection.Type and its name,
// used for identifying and mapping struct fields during conversion.
type typeNamePair struct {
	fieldType reflect.Type
	fieldName string
}

// DebugLogger allows you to get debugging messages if necessary.
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

// Converter knows how to convert one type to another. It manages registered
// conversion functions and provides mechanisms for copying fields between structs.
type Converter struct {
	// funcs maps a typePair (source and destination types) to a reflect.Value
	// representing a custom conversion function. These functions handle specific
	// type-to-type conversions.
	funcs map[typePair]reflect.Value

	// structFieldDests maps a source field's type and name (typeNamePair) to a list
	// of destination field type and name (typeNamePair) where its value should be copied.
	// This supports copying values from a source struct field to one or more destination struct fields.
	structFieldDests map[typeNamePair][]typeNamePair

	// structFieldSources maps a destination field's type and name (typeNamePair) to a list
	// of potential source field type and name (typeNamePair). This allows for reverse lookup
	// during conversion when the 'DestFromSource' flag is set, enabling copying from
	// a source field that matches a given destination field.
	structFieldSources map[typeNamePair][]typeNamePair

	// Debug, if non-nil, will be called to print helpful debugging information during conversion.
	// This can be quite verbose and is primarily for diagnostic purposes.
	Debug DebugLogger

	// NameFunc is a function used to retrieve the name of a reflect.Type. This name is crucial
	// for determining type compatibility during conversion. By default, it returns the standard Go type name.
	NameFunc func(t reflect.Type) string
}

// NewConverter creates and returns a new initialized Converter object.
// It sets up the internal maps and defaults the NameFunc to use reflect.Type.Name().
func NewConverter() *Converter {
	return &Converter{
		funcs:              map[typePair]reflect.Value{},
		NameFunc:           func(t reflect.Type) string { return t.Name() },
		structFieldDests:   map[typeNamePair][]typeNamePair{},
		structFieldSources: map[typeNamePair][]typeNamePair{},
	}
}

// Scope is passed to custom conversion functions (registered via Register) to allow them
// to continue an ongoing conversion for sub-objects. This ensures that nested conversions
// use the same Converter instance and flags. If multiple Converters are in use, Scope
// provides the context of the currently active Converter.
type Scope interface {
	// Convert attempts to convert 'src' into 'dest' within the current conversion scope.
	// It's typically used by custom conversion functions to handle nested or related objects.
	// Calling it with identical 'src' and 'dest' as the parent call will lead to infinite recursion.
	Convert(src, dest interface{}, flags FieldMatchingFlags) error

	// SrcTag returns the reflect.StructTag of the struct field from which the current source item originated.
	// If the source was not part of a struct field (e.g., a top-level object), it will return an empty tag.
	SrcTag() reflect.StructTag

	// DestTag returns the reflect.StructTag of the struct field into which the current destination item is being converted.
	// If the destination is not part of a struct field, it will return an empty tag.
	DestTag() reflect.StructTag

	// Flags returns the FieldMatchingFlags that were originally set for the current conversion operation.
	Flags() FieldMatchingFlags

	// Meta returns any additional metadata object that was originally passed to the top-level Convert call.
	Meta() *Meta
}

// Meta is an optional object that can be supplied to the top-level Convert call,
// providing additional context or parameters to custom conversion functions.
// It is typically used by a Scheme (a higher-level conversion orchestrator).
type Meta struct {
	SrcVersion  string
	DestVersion string

	// TODO: If needed, add a user data field here.
}

// scope contains information about an ongoing conversion, such as the Converter instance,
// flags, and metadata. It also tracks the source and destination reflection stacks to
// manage struct field tags during recursive conversions.
type scope struct {
	converter *Converter
	meta      *Meta
	flags     FieldMatchingFlags

	// srcStack and destStack store elements (tags and values) from the source and destination
	// reflection paths. They are kept separate because the mapping between source and destination
	// fields/types may not be one-to-one during complex conversions.
	srcStack  scopeStack
	destStack scopeStack
}

// scopeStackElem represents a single element in the conversion stack, holding a struct tag
// and its associated reflection.Value.
type scopeStackElem struct {
	tag   reflect.StructTag
	value reflect.Value
}

// scopeStack is a custom slice type used to manage the stack of conversion elements.
type scopeStack []scopeStackElem

// pop removes the top element from the scopeStack.
func (s *scopeStack) pop() {
	n := len(*s)
	*s = (*s)[:n-1]
}

// push adds a new element to the top of the scopeStack.
func (s *scopeStack) push(e scopeStackElem) {
	*s = append(*s, e)
}

// top returns a pointer to the top element of the scopeStack.
func (s *scopeStack) top() *scopeStackElem {
	return &(*s)[len(*s)-1]
}

// Convert continues a conversion within the current scope, delegating to the underlying Converter.
// This allows custom conversion functions to trigger further conversions.
func (s *scope) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.Convert(src, dest, flags, s.meta)
}

// SrcTag returns the tag of the struct field associated with the current source element on the stack.
func (s *scope) SrcTag() reflect.StructTag {
	return s.srcStack.top().tag
}

// DestTag returns the tag of the struct field associated with the current destination element on the stack.
func (s *scope) DestTag() reflect.StructTag {
	return s.destStack.top().tag
}

// Flags returns the FieldMatchingFlags with which the current conversion was initiated.
func (s *scope) Flags() FieldMatchingFlags {
	return s.flags
}

// Meta returns the Meta object that was passed at the beginning of the top-level Convert call.
func (s *scope) Meta() *Meta {
	return s.meta
}

// Register registers a custom conversion function with the Converter. The conversionFunc must adhere
// to a specific signature: it must take three parameters (a pointer to the input type, a pointer
// to the output type, and a conversion.Scope interface) and must return an error.
// This function performs strict type checking on the provided conversionFunc.
//
// Example:
// c.Register(func(in *Pod, out *v1beta1.Pod, s Scope) error { ... return nil })
func (c *Converter) Register(conversionFunc interface{}) error {
	fv := reflect.ValueOf(conversionFunc)
	ft := fv.Type()
	// Block Logic: Validate that conversionFunc is indeed a function.
	if ft.Kind() != reflect.Func {
		return fmt.Errorf("expected func, got: %v", ft)
	}
	// Block Logic: Validate that the function has exactly three input parameters.
	if ft.NumIn() != 3 {
		return fmt.Errorf("expected three 'in' params, got: %v", ft)
	}
	// Block Logic: Validate that the function has exactly one output parameter.
	if ft.NumOut() != 1 {
		return fmt.Errorf("expected one 'out' param, got: %v", ft)
	}
	// Block Logic: Validate that the first input parameter is a pointer type.
	if ft.In(0).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 0, got: %v", ft)
	}
	// Block Logic: Validate that the second input parameter is a pointer type.
	if ft.In(1).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 1, got: %v", ft)
	}
	// Block Logic: Validate that the third input parameter matches the conversion.Scope interface type.
	scopeType := Scope(nil)
	if e, a := reflect.TypeOf(&scopeType).Elem(), ft.In(2); e != a {
		return fmt.Errorf("expected '%v' arg for 'in' param 2, got '%v' (%v)", e, a, ft)
	}
	var forErrorType error
	// This convolution is necessary, otherwise TypeOf picks up on the fact
	// that forErrorType is nil.
	errorType := reflect.TypeOf(&forErrorType).Elem()
	// Block Logic: Validate that the single output parameter is of type error.
	if ft.Out(0) != errorType {
		return fmt.Errorf("expected error return, got: %v", ft)
	}
	// If all validations pass, store the conversion function in the 'funcs' map
	// using the element types of the input and output pointers as the key.
	c.funcs[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	return nil
}

// SetStructFieldCopy registers a correspondence between a source struct field and a
// destination struct field for copying purposes. When a field matching srcFieldType
// and srcFieldName is encountered in a source struct, its value will be copied into
// the corresponding field in the destination struct that matches destFieldType and destFieldName.
// This function can be called multiple times to define various field copy rules.
func (c *Converter) SetStructFieldCopy(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	st := reflect.TypeOf(srcFieldType)
	dt := reflect.TypeOf(destFieldType)
	srcKey := typeNamePair{st, srcFieldName}
	destKey := typeNamePair{dt, destFieldName}
	// Add the destination key to the list of destinations for the given source key.
	c.structFieldDests[srcKey] = append(c.structFieldDests[srcKey], destKey)
	// Add the source key to the list of sources for the given destination key (for reverse lookup).
	c.structFieldSources[destKey] = append(c.structFieldSources[destKey], srcKey)
	return nil
}

// FieldMatchingFlags defines a set of flags that control how struct fields are
// matched and copied during the conversion process. These flags can be combined
// using bitwise OR operations.
type FieldMatchingFlags int

const (
	// DestFromSource is the default behavior (value 0). It instructs the Converter
	// to iterate through the destination struct's fields and attempt to find a matching
	// source field to copy from. Source fields that do not have a corresponding
	// destination field will be ignored. This flag is implicitly active if no other
	// field matching flags are specified, or if SourceToDest is not set.
	DestFromSource FieldMatchingFlags = 0
	// SourceToDest instructs the Converter to iterate through the source struct's fields
	// and attempt to copy them into matching destination fields. Destination fields
	// that do not have a corresponding source field will be ignored.
	SourceToDest FieldMatchingFlags = 1 << iota
	// IgnoreMissingFields prevents the Converter from returning an error if a corresponding
	// source or destination field (depending on the primary matching direction) cannot be found.
	// This allows for partial conversions without strict field presence requirements.
	IgnoreMissingFields
	// AllowDifferentFieldTypeNames relaxes the type matching criteria, permitting conversion
	// between fields whose underlying types have different names, as long as they are
	// otherwise compatible (e.g., same kind).
	AllowDifferentFieldTypeNames
)

// IsSet returns true if the specified flag (or combination of flags) is set within
// the current FieldMatchingFlags value. Special handling is provided for DestFromSource,
// which is the default when SourceToDest is not set.
func (f FieldMatchingFlags) IsSet(flag FieldMatchingFlags) bool {
	// Block Logic: Special handling for DestFromSource.
	// DestFromSource is the default (0), meaning it's set if SourceToDest is NOT set.
	if flag == DestFromSource {
		return f&SourceToDest != SourceToDest
	}
	// For other flags, check if the bit is set using a bitwise AND operation.
	return f&flag == flag
}

// Convert will translate src to dest if a conversion function is registered or
// if the default copying mechanism can handle the type pair. Both 'src' and 'dest'
// must be pointers to the actual data structures. If no appropriate conversion is
// found, an error is returned.
// The 'flags' parameter controls the field matching behavior for struct conversions.
// The 'meta' object allows passing arbitrary context to custom conversion functions.
// Note: This function is not safe for objects with cyclic references as it may lead to infinite recursion.
func (c *Converter) Convert(src, dest interface{}, flags FieldMatchingFlags, meta *Meta) error {
	// Block Logic: Ensure 'dest' is a settable pointer and get its reflect.Value.
	dv, err := EnforcePtr(dest)
	if err != nil {
		return err
	}
	if !dv.CanAddr() {
		return fmt.Errorf("can't write to dest")
	}
	// Block Logic: Ensure 'src' is a valid pointer and get its reflect.Value.
	sv, err := EnforcePtr(src)
	if err != nil {
		return err
	}
	// Block Logic: Initialize a new scope for the current conversion operation.
	s := &scope{
		converter: c,
		flags:     flags,
		meta:      meta,
	}
	// Block Logic: Push empty scope elements onto the stacks.
	// This ensures that `top()` calls for struct tags never panic, even if no actual
	// struct fields are being processed (e.g., top-level primitive type conversion).
	s.srcStack.push(scopeStackElem{})
	s.destStack.push(scopeStackElem{})
	// Delegate the actual conversion to the recursive 'convert' method.
	return c.convert(sv, dv, s)
}

// convert recursively copies the value from 'sv' (source reflect.Value) into 'dv' (destination reflect.Value).
// It prioritizes registered custom conversion functions. If no custom function is found, it attempts
// direct assignment, type-safe conversion, or recursive conversion for complex types (structs, slices, pointers, maps).
func (c *Converter) convert(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()
	// Block Logic: Check if a custom conversion function is registered for this source-destination type pair.
	if fv, ok := c.funcs[typePair{st, dt}]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Calling custom conversion of '%v' to '%v'", st, dt)
		}
		// Call the custom conversion function with the source address, destination address, and current scope.
		args := []reflect.Value{sv.Addr(), dv.Addr(), reflect.ValueOf(scope)}
		ret := fv.Call(args)[0].Interface()
		// Handle potential nil error return from the custom function.
		if ret == nil {
			return nil
		}
		return ret.(error)
	}

	// Block Logic: If 'AllowDifferentFieldTypeNames' flag is not set, enforce matching type names.
	if !scope.flags.IsSet(AllowDifferentFieldTypeNames) && c.NameFunc(dt) != c.NameFunc(st) {
		return fmt.Errorf("can't convert %v to %v because type names don't match (%v, %v).", st, dt, c.NameFunc(st), c.NameFunc(dt))
	}

	// Block Logic: Attempt direct assignment for simple types.
	if st.AssignableTo(dt) {
		dv.Set(sv)
		return nil
	}
	// Block Logic: Attempt type-safe conversion (e.g., int to float) for compatible types.
	if st.ConvertibleTo(dt) {
		dv.Set(sv.Convert(dt))
		return nil
	}

	if c.Debug != nil {
		c.Debug.Logf("Trying to convert '%v' to '%v'", st, dt)
	}

	// Block Logic: Push current source and destination values onto their respective stacks.
	// These values (and their associated tags if applicable) provide context for nested conversions.
	scope.srcStack.push(scopeStackElem{value: sv})
	scope.destStack.push(scopeStackElem{value: dv})
	// Ensure that stack elements are popped when the current conversion frame exits.
	defer scope.srcStack.pop()
	defer scope.destStack.pop()

	// Block Logic: Handle conversion based on the Kind of the destination value.
	switch dv.Kind() {
	case reflect.Struct:
		// For structs, delegate to convertStruct for field-by-field conversion.
		return c.convertStruct(sv, dv, scope)
	case reflect.Slice:
		// If the source slice is nil, set the destination slice to its zero value (nil).
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt))
			return nil
		}
		// Create a new destination slice with the same length and capacity as the source.
		dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
		// Recursively convert each element from the source slice to the destination slice.
		for i := 0; i < sv.Len(); i++ {
			if err := c.convert(sv.Index(i), dv.Index(i), scope); err != nil {
				return err
			}
		}
	case reflect.Ptr:
		// If the source pointer is nil, set the destination pointer to its zero value (nil).
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt))
			return nil
		}
		// Create a new value for the element that the destination pointer points to.
		dv.Set(reflect.New(dt.Elem()))
		// Recursively convert the element pointed to by the source pointer to the element
		// pointed to by the new destination pointer.
		return c.convert(sv.Elem(), dv.Elem(), scope)
	case reflect.Map:
		// If the source map is nil, set the destination map to its zero value (nil).
		if sv.IsNil() {
			dv.Set(reflect.Zero(dt))
			return nil
		}
		// Create a new destination map.
		dv.Set(reflect.MakeMap(dt))
		// Iterate over each key in the source map.
		for _, sk := range sv.MapKeys() {
			// Create new destination key and value elements.
			dk := reflect.New(dt.Key()).Elem()
			// Recursively convert the source key to the destination key type.
			if err := c.convert(sk, dk, scope); err != nil {
				return err
			}
			dkv := reflect.New(dt.Elem()).Elem()
			// Recursively convert the source map value to the destination value type.
			if err := c.convert(sv.MapIndex(sk), dkv, scope); err != nil {
				return err
			}
			// Set the converted key-value pair in the destination map.
			dv.SetMapIndex(dk, dkv)
		}
	default:
		// If the type kind is not handled, return an error.
		return fmt.Errorf("couldn't copy '%v' into '%v'", st, dt)
	}
	return nil
}

// convertStruct handles the field-by-field conversion between two structs.
// It iterates through the fields of either the source or destination struct (depending
// on the FieldMatchingFlags) and attempts to copy values between matching fields.
func (c *Converter) convertStruct(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	listType := dt
	// Block Logic: Determine which struct's fields to iterate over based on the flags.
	// If SourceToDest is set, iterate over source fields; otherwise, iterate over destination fields.
	if scope.flags.IsSet(SourceToDest) {
		listType = st
	}
	// Block Logic: Iterate through each field of the chosen struct type.
	for i := 0; i < listType.NumField(); i++ {
		f := listType.Field(i)
		// Block Logic: Check if a custom field copying rule applies to this field.
		if found, err := c.checkStructField(f.Name, sv, dv, scope); found {
			if err != nil {
				return err
			}
			continue // If a custom rule handled it, move to the next field.
		}
		// Get the destination and source field values by name.
		df := dv.FieldByName(f.Name)
		sf := sv.FieldByName(f.Name)
		// Block Logic: Update struct tags on the scope stacks for the current source and destination fields.
		if sf.IsValid() {
			field, _ := st.FieldByName(f.Name) // Error check not needed as sf.IsValid() ensures field exists.
			scope.srcStack.top().tag = field.Tag
		}
		if df.IsValid() {
			field, _ := dt.FieldByName(f.Name) // Error check not needed as df.IsValid() ensures field exists.
			scope.destStack.top().tag = field.Tag
		}
		// TODO: The comment "set top level of scope.src/destTagStack with these field tags here."
		// indicates a potential area for improvement or confirmation of behavior.

		// Block Logic: Handle cases where either the destination or source field is not valid.
		if !df.IsValid() || !sf.IsValid() {
			switch {
			case scope.flags.IsSet(IgnoreMissingFields):
				// If IgnoreMissingFields is set, simply continue without an error.
			case scope.flags.IsSet(SourceToDest):
				// If SourceToDest, and destination field is missing, report error.
				return fmt.Errorf("%v not present in dest (%v to %v)", f.Name, st, dt)
			default:
				// Default (DestFromSource), if source field is missing, report error.
				return fmt.Errorf("%v not present in src (%v to %v)", f.Name, st, dt)
			}
			continue // Move to the next field if handled or ignored.
		}
		// Block Logic: Recursively convert the source field's value into the destination field's value.
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

// checkStructField determines if a given fieldName matches any registered struct field
// copying rules and performs the copy if applicable. It returns true if a replacement
// was made (or an error occurred during an attempted replacement), and an error if the
// conversion failed. If no matching rule is found, it returns false and a nil error.
func (c *Converter) checkStructField(fieldName string, sv, dv reflect.Value, scope *scope) (bool, error) {
	replacementMade := false
	// Block Logic: Handle 'DestFromSource' flag behavior.
	if scope.flags.IsSet(DestFromSource) {
		df := dv.FieldByName(fieldName)
		if !df.IsValid() {
			return false, nil // Destination field not found, no replacement possible this way.
		}
		destKey := typeNamePair{df.Type(), fieldName}
		// Iterate through potential source field keys registered for this destination key.
		for _, potentialSourceKey := range c.structFieldSources[destKey] {
			sf := sv.FieldByName(potentialSourceKey.fieldName)
			// Check if the source field exists and its type matches the potential source key's type.
			if !sf.IsValid() {
				continue // Source field not found, try next potential source.
			}
			if sf.Type() == potentialSourceKey.fieldType {
				// Both name and type matched: perform the conversion/copy.
				if err := c.convert(sf, df, scope); err != nil {
					return true, err // Return true and error, indicating an attempt was made and failed.
				}
				replacementMade = true // Mark that a replacement was successfully made.
			}
		}
		return replacementMade, nil // Return whether any replacement was made.
	}

	// Block Logic: Handle 'SourceToDest' flag behavior (or default if no flags imply DestFromSource).
	sf := sv.FieldByName(fieldName)
	if !sf.IsValid() {
		return false, nil // Source field not found, no replacement possible this way.
	}
	srcKey := typeNamePair{sf.Type(), fieldName}
	// Iterate through potential destination field keys registered for this source key.
	for _, potentialDestKey := range c.structFieldDests[srcKey] {
		df := dv.FieldByName(potentialDestKey.fieldName)
		// Check if the destination field exists and its type matches the potential destination key's type.
		if !df.IsValid() {
			continue // Destination field not found, try next potential destination.
		}
		if df.Type() == potentialDestKey.fieldType {
			// Both name and type matched: perform the conversion/copy.
			if err := c.convert(sf, df, scope); err != nil {
				return true, err // Return true and error, indicating an attempt was made and failed.
			}
			replacementMade = true // Mark that a replacement was successfully made.
		}
	}
	return replacementMade, nil // Return whether any replacement was made.
}
