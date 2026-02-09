/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.instrumentation.impl;

import org.elasticsearch.core.Strings;
import org.elasticsearch.entitlement.instrumentation.CheckMethod;
import org.elasticsearch.entitlement.instrumentation.EntitlementInstrumented;
import org.elasticsearch.entitlement.instrumentation.Instrumenter;
import org.elasticsearch.entitlement.instrumentation.MethodKey;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.RecordComponentVisitor;
import org.objectweb.asm.Type;
import org.objectweb.asm.util.CheckClassAdapter;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Map;
import java.util.stream.Stream;

import static org.objectweb.asm.ClassWriter.COMPUTE_FRAMES;
import static org.objectweb.asm.ClassWriter.COMPUTE_MAXS;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.CHECKCAST;
import static org.objectweb.asm.Opcodes.INVOKEINTERFACE;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;

/**
 * @brief Implements bytecode instrumentation to inject entitlement checks into methods.
 *
 * This class leverages the ASM library to read, modify, and write Java bytecode.
 * It identifies methods that need instrumentation (those not already annotated with
 * {@link EntitlementInstrumented}) and injects calls to an entitlement checker
 * at the beginning of these methods. The instrumentation process involves:
 * 1. Identifying target methods based on a provided map of {@link MethodKey}s to {@link CheckMethod}s.
 * 2. Inserting bytecode instructions to retrieve an instance of the entitlement checker.
 * 3. Pushing relevant arguments (caller class, method arguments) onto the stack.
 * 4. Invoking the appropriate entitlement check method.
 * 5. Optionally adding an {@link EntitlementInstrumented} annotation to the class
 *    to prevent redundant instrumentation.
 *
 * The instrumentation is designed to be pluggable, allowing different entitlement
 * logic to be applied without modifying the original source code. It also includes
 * bytecode verification steps to ensure the integrity of the modified classes.
 *
 * Algorithm: Bytecode transformation using ASM ClassVisitor and MethodVisitor for method injection.
 * Time Complexity: O(C * M * I) where C is the number of classes, M is the number of methods per class,
 *                  and I is the average number of instructions in a method being analyzed/modified.
 * Space Complexity: O(B) where B is the size of the bytecode being processed.
 */
public final class InstrumenterImpl implements Instrumenter {
    private static final Logger logger = LogManager.getLogger(InstrumenterImpl.class);

    /**
     * @brief The descriptor for the method used to obtain the entitlement checker class instance.
     * Functional Utility: Specifies the signature of the static method to call for retrieving the singleton
     *                     instance of the checker class, crucial for dynamically invoking entitlement logic.
     */
    private final String getCheckerClassMethodDescriptor;
    /**
     * @brief The internal name (ASM format) of the class that provides the entitlement checker instance.
     * Functional Utility: Used during bytecode generation to correctly reference the class containing
     *                     the `instance` method.
     */
    private final String handleClass;

    /**
     * @brief A suffix appended to class names during testing.
     * Functional Utility: Prevents class name collisions when running tests without an agent that
     *                     replaces classes in-place, allowing multiple versions of an instrumented
     *                     class to coexist in the same ClassLoader.
     */
    private final String classNameSuffix;
    /**
     * @brief A map of {@link MethodKey} to {@link CheckMethod} instances.
     * Functional Utility: Defines which methods require entitlement checks and specifies the
     *                     corresponding checker method to invoke. This map drives the instrumentation logic.
     */
    private final Map<MethodKey, CheckMethod> checkMethods;

    /**
     * @brief Constructs an InstrumenterImpl instance.
     * @param handleClass The internal name of the class providing the entitlement checker instance.
     * @param getCheckerClassMethodDescriptor The method descriptor for obtaining the checker class instance.
     * @param classNameSuffix A suffix to append to instrumented class names (primarily for testing).
     * @param checkMethods A map defining which methods to instrument and with which check methods.
     * Functional Utility: Initializes the core components required for performing bytecode instrumentation.
     */
    InstrumenterImpl(
        String handleClass,
        String getCheckerClassMethodDescriptor,
        String classNameSuffix,
        Map<MethodKey, CheckMethod> checkMethods
    ) {
        this.handleClass = handleClass;
        this.getCheckerClassMethodDescriptor = getCheckerClassMethodDescriptor;
        this.classNameSuffix = classNameSuffix;
        this.checkMethods = checkMethods;
    }

    /**
     * @brief Factory method to create an {@link InstrumenterImpl} instance.
     * @param checkerClass The class that provides the entitlement checking logic.
     * @param checkMethods A map defining which methods to instrument and with which check methods.
     * @return A new {@link InstrumenterImpl} instance.
     * Functional Utility: Simplifies the creation of the instrumenter by deriving necessary
     *                     parameters (like `handleClass` and `getCheckerClassMethodDescriptor`)
     *                     from the provided `checkerClass`.
     */
    public static InstrumenterImpl create(Class<?> checkerClass, Map<MethodKey, CheckMethod> checkMethods) {

        Type checkerClassType = Type.getType(checkerClass);
        String handleClass = checkerClassType.getInternalName() + "Handle";
        String getCheckerClassMethodDescriptor = Type.getMethodDescriptor(checkerClassType);
        return new InstrumenterImpl(handleClass, getCheckerClassMethodDescriptor, "", checkMethods);
    }

    /**
     * @brief Retrieves the bytecode of a given class from the classpath.
     * @param clazz The {@link Class} object to retrieve bytecode for.
     * @return A {@link ClassFileInfo} containing the class file name and its bytecode.
     * @throws IOException if an I/O error occurs while reading the class stream.
     * @throws IllegalStateException if the class file cannot be found in the classpath.
     * Functional Utility: Provides the raw bytecode necessary for instrumentation.
     */
    static ClassFileInfo getClassFileInfo(Class<?> clazz) throws IOException {
        String internalName = Type.getInternalName(clazz);
        String fileName = "/" + internalName + ".class";
        byte[] originalBytecodes;
        /**
         * Block Logic: Attempts to open an input stream to read the class file's bytecode.
         * Functional Utility: Safely acquires and manages the `InputStream` for reading the class's binary data,
         *                     ensuring it's closed automatically after use.
         * Pre-condition: `fileName` is the expected path to the class file within the classpath.
         */
        try (InputStream classStream = clazz.getResourceAsStream(fileName)) {
            /**
             * Block Logic: Checks if the class file stream was successfully opened.
             * Functional Utility: Throws an {@link IllegalStateException} if the class file
             *                     is not found, preventing further processing with missing data.
             * Pre-condition: `classStream` is the result of `clazz.getResourceAsStream(fileName)`.
             */
            if (classStream == null) {
                throw new IllegalStateException("Classfile not found in jar: " + fileName);
            }
            originalBytecodes = classStream.readAllBytes();
        }
        return new ClassFileInfo(fileName, originalBytecodes);
    }

    /**
     * @brief Represents the phase of bytecode verification during instrumentation.
     * Functional Utility: Provides clear semantic distinction for logging and debugging
     *                     when bytecode is verified before and after modifications.
     */
    private enum VerificationPhase {
        /**
         * @brief Indicates verification of bytecode *before* instrumentation has been applied.
         */
        BEFORE_INSTRUMENTATION,
        /**
         * @brief Indicates verification of bytecode *after* instrumentation has been applied.
         */
        AFTER_INSTRUMENTATION
    }

    /**
     * @brief Performs bytecode verification on the given classfile buffer using ASM's {@link CheckClassAdapter}.
     * @param classfileBuffer The bytecode of the class to verify.
     * @return A {@link String} containing verification errors, or an empty string if verification passes.
     * Functional Utility: Ensures that the bytecode is structurally and semantically valid according to JVM specifications,
     *                     catching potential issues introduced during or before instrumentation.
     */
    private static String verify(byte[] classfileBuffer) {
        ClassReader reader = new ClassReader(classfileBuffer);
        StringWriter stringWriter = new StringWriter();
        PrintWriter printWriter = new PrintWriter(stringWriter);
        CheckClassAdapter.verify(reader, false, printWriter);
        return stringWriter.toString();
    }

    /**
     * @brief Performs bytecode verification and logs the outcome (success, failure, or inconclusive).
     * @param classfileBuffer The bytecode of the class to verify.
     * @param className The name of the class being verified, used for logging.
     * @param phase The {@link VerificationPhase} indicating when the verification is performed.
     * Functional Utility: Provides crucial debugging information by validating bytecode at different stages
     *                     of instrumentation, helping to identify and diagnose bytecode manipulation errors.
     */
    private static void verifyAndLog(byte[] classfileBuffer, String className, VerificationPhase phase) {
        /**
         * Block Logic: Attempts to verify the bytecode and log the results.
         * Functional Utility: Encapsulates the verification process, handling potential exceptions
         *                     that might occur during bytecode analysis.
         */
        try {
            String result = verify(classfileBuffer);
            if (result.isEmpty() == false) {
                logger.error(Strings.format("Bytecode verification (%s) for class [%s] failed: %s", phase, className, result));
            } else {
                logger.info("Bytecode verification ({}) for class [{}] passed", phase, className);
            }
        } catch (ClassCircularityError e) {
            /**
             * Block Logic: Catches {@link ClassCircularityError} which can occur during complex bytecode verification.
             * Functional Utility: Treats this specific error as an "inconclusive" verification rather than a hard failure,
             *                     acknowledging the challenges of bytecode verification in certain scenarios.
             * Pre-condition: A {@link ClassCircularityError} is thrown during bytecode verification.
             * Invariant: The error is logged as a warning, and the verification is considered inconclusive.
             */
            // Apparently, verification during instrumentation is challenging for class resolution and loading
            // Treat this not as an error, but as "inconclusive"
            logger.warn(Strings.format("Cannot perform bytecode verification (%s) for class [%s]", phase, className), e);
        } catch (IllegalArgumentException e) {
            /**
             * Block Logic: Catches {@link IllegalArgumentException} which can be thrown by ASM's CheckClassAdapter.
             * Functional Utility: Logs this exception as a bytecode verification failure, providing insights
             *                     into potential issues with the class structure or the instrumentation process.
             * Pre-condition: An {@link IllegalArgumentException} is thrown by the bytecode verifier.
             * Invariant: The exception is logged as an error, indicating a verification failure.
             */
            // The ASM CheckClassAdapter in some cases throws this instead of printing the error
            logger.error(Strings.format("Bytecode verification (%s) for class [%s] failed", phase, className), e);
        }
    }

    @Override
    public byte[] instrumentClass(String className, byte[] classfileBuffer, boolean verify) {
        /**
         * Block Logic: Conditionally performs bytecode verification before instrumentation.
         * Functional Utility: Ensures the input bytecode is valid prior to modification,
         *                     helping to isolate issues and validate the source class.
         * Pre-condition: `verify` flag is true.
         */
        if (verify) {
            verifyAndLog(classfileBuffer, className, VerificationPhase.BEFORE_INSTRUMENTATION);
        }

        ClassReader reader = new ClassReader(classfileBuffer);
        ClassWriter writer = new ClassWriter(reader, COMPUTE_FRAMES | COMPUTE_MAXS);
        ClassVisitor visitor = new EntitlementClassVisitor(Opcodes.ASM9, writer, className);
        reader.accept(visitor, 0);
        var outBytes = writer.toByteArray();

        /**
         * Block Logic: Conditionally performs bytecode verification after instrumentation.
         * Functional Utility: Validates the correctness of the generated (instrumented) bytecode,
         *                     catching any structural or semantic errors introduced during the instrumentation process.
         * Pre-condition: `verify` flag is true.
         */
        if (verify) {
            verifyAndLog(outBytes, className, VerificationPhase.AFTER_INSTRUMENTATION);
        }

        return outBytes;
    }

    /**
     * @brief A custom {@link ClassVisitor} that intercepts class elements to apply entitlement instrumentation.
     * Functional Utility: This visitor is central to the bytecode modification process, responsible for:
     *                     1. Detecting the {@link EntitlementInstrumented} annotation.
     *                     2. Conditionally adding the {@link EntitlementInstrumented} annotation if not present.
     *                     3. Delegating to {@link EntitlementMethodVisitor} for method-level instrumentation.
     *                     4. Handling class name transformations for testing.
     *
     * It ensures that only classes targeted for instrumentation are modified and that redundant
     * instrumentation is avoided.
     */
    class EntitlementClassVisitor extends ClassVisitor {

        /**
         * @brief The descriptor string for the {@link EntitlementInstrumented} annotation.
         * Functional Utility: Used to identify if a class or method is already marked for entitlement instrumentation.
         */
        private static final String ENTITLEMENT_ANNOTATION_DESCRIPTOR = Type.getDescriptor(EntitlementInstrumented.class);

        /**
         * @brief The internal name (ASM format) of the class being visited.
         * Functional Utility: Provides context for logging and for identifying methods within this specific class.
         */
        private final String className;

        /**
         * @brief Flag indicating if the {@link EntitlementInstrumented} annotation is already present on the class.
         * Functional Utility: Prevents redundant annotation additions.
         */
        private boolean isAnnotationPresent;
        /**
         * @brief Flag indicating if the {@link EntitlementInstrumented} annotation needs to be added to the class.
         * Functional Utility: Controls the conditional addition of the annotation to the class.
         */
        private boolean annotationNeeded = true;

        /**
         * @brief Constructs a new EntitlementClassVisitor.
         * @param api The ASM API version being used.
         * @param classVisitor The next {@link ClassVisitor} in the chain.
         * @param className The internal name of the class being visited.
         * Functional Utility: Initializes the visitor with necessary context for traversing and modifying the class structure.
         */
        EntitlementClassVisitor(int api, ClassVisitor classVisitor, String className) {
            super(api, classVisitor);
            this.className = className;
        }

        /**
         * @brief Visits the header of the class.
         * @param version The class file format version.
         * @param access The class's access flags.
         * @param name The internal name of the class.
         * @param signature The signature of the class.
         * @param superName The internal name of the superclass.
         * @param interfaces The internal names of the implemented interfaces.
         * Functional Utility: Modifies the class's internal name if a `classNameSuffix` is configured, typically for testing.
         */
        @Override
        public void visit(int version, int access, String name, String signature, String superName, String[] interfaces) {
            super.visit(version, access, name + classNameSuffix, signature, superName, interfaces);
        }

        /**
         * @brief Visits an annotation of the class.
         * @param descriptor The class descriptor of the annotation class.
         * @param visible `true` if the annotation is visible at runtime.
         * @return An {@link AnnotationVisitor} to visit the annotation's values.
         * Functional Utility: Detects if the {@link EntitlementInstrumented} annotation is already present
         *                     on the class, which influences whether it needs to be added later.
         */
        @Override
        public AnnotationVisitor visitAnnotation(String descriptor, boolean visible) {
            /**
             * Block Logic: Checks if the visited annotation is {@link EntitlementInstrumented} and is visible at runtime.
             * Functional Utility: Sets flags to indicate that the class is already annotated,
             *                     preventing redundant annotation addition.
             */
            if (visible && descriptor.equals(ENTITLEMENT_ANNOTATION_DESCRIPTOR)) {
                isAnnotationPresent = true;
                annotationNeeded = false;
            }
            return cv.visitAnnotation(descriptor, visible);
        }

        /**
         * @brief Visits an inner class or a nest member.
         * Functional Utility: Ensures that the class being instrumented has the {@link EntitlementInstrumented}
         *                     annotation added before processing inner classes or nest members, if needed.
         */
        @Override
        public void visitNestMember(String nestMember) {
            addClassAnnotationIfNeeded();
            super.visitNestMember(nestMember);
        }

        /**
         * @brief Visits a permitted subclass (for sealed classes).
         * Functional Utility: Ensures that the class being instrumented has the {@link EntitlementInstrumented}
         *                     annotation added before processing permitted subclasses, if needed.
         */
        @Override
        public void visitPermittedSubclass(String permittedSubclass) {
            addClassAnnotationIfNeeded();
            super.visitPermittedSubclass(permittedSubclass);
        }

        /**
         * @brief Visits an inner class.
         * Functional Utility: Ensures that the class being instrumented has the {@link EntitlementInstrumented}
         *                     annotation added before processing inner class details, if needed.
         */
        @Override
        public void visitInnerClass(String name, String outerName, String innerName, int access) {
            addClassAnnotationIfNeeded();
            super.visitInnerClass(name, outerName, innerName, access);
        }

        /**
         * @brief Visits a field of the class.
         * Functional Utility: Ensures that the class being instrumented has the {@link EntitlementInstrumented}
         *                     annotation added before processing field details, if needed.
         */
        @Override
        public FieldVisitor visitField(int access, String name, String descriptor, String signature, Object value) {
            addClassAnnotationIfNeeded();
            return super.visitField(access, name, descriptor, signature, value);
        }

        /**
         * @brief Visits a record component.
         * Functional Utility: Ensures that the class being instrumented has the {@link EntitlementInstrumented}
         *                     annotation added before processing record component details, if needed.
         */
        @Override
        public RecordComponentVisitor visitRecordComponent(String name, String descriptor, String signature) {
            addClassAnnotationIfNeeded();
            return super.visitRecordComponent(name, descriptor, signature);
        }

        /**
         * @brief Visits a method of the class.
         * @param access The method's access flags.
         * @param name The method's name.
         * @param descriptor The method's descriptor.
         * @param signature The method's signature.
         * @param exceptions The internal names of the method's exceptions.
         * @return A {@link MethodVisitor} to visit the method's code, or `null` if the method should not be visited.
         * Functional Utility: Determines whether a method requires instrumentation and wraps its {@link MethodVisitor}
         *                     with an {@link EntitlementMethodVisitor} if an entitlement check is needed.
         */
        @Override
        public MethodVisitor visitMethod(int access, String name, String descriptor, String signature, String[] exceptions) {
            addClassAnnotationIfNeeded();
            var mv = super.visitMethod(access, name, descriptor, signature, exceptions);
            /**
             * Block Logic: Checks if the class is not already annotated with {@link EntitlementInstrumented}.
             * Functional Utility: Only attempts to instrument methods if the entire class hasn't been
             *                     marked as instrumented, preventing redundant checks.
             */
            if (isAnnotationPresent == false) {
                boolean isStatic = (access & ACC_STATIC) != 0;
                boolean isCtor = "<init>".equals(name);
                var key = new MethodKey(className, name, Stream.of(Type.getArgumentTypes(descriptor)).map(Type::getInternalName).toList());
                var instrumentationMethod = checkMethods.get(key);
                /**
                 * Block Logic: Determines if the current method should be instrumented based on the `checkMethods` map.
                 * Functional Utility: If a matching `instrumentationMethod` is found, the method's visitor is wrapped
                 *                     with `EntitlementMethodVisitor` to inject the entitlement check. Otherwise,
                 *                     it proceeds without modification.
                 */
                if (instrumentationMethod != null) {
                    logger.debug("Will instrument {}", key);
                    return new EntitlementMethodVisitor(Opcodes.ASM9, mv, isStatic, isCtor, descriptor, instrumentationMethod);
                } else {
                    logger.trace("Will not instrument {}", key);
                }
            }
            return mv;
        }

        /**
         * @brief Conditionally adds the {@link EntitlementInstrumented} annotation to the class if it hasn't been added yet.
         *
         * A class annotation can be added via visitAnnotation; we need to call visitAnnotation after all other visitAnnotation
         * calls (in case one of them detects our annotation is already present), but before any other subsequent visit* method is called
         * (up to visitMethod -- if no visitMethod is called, there is nothing to instrument).
         * This includes visitNestMember, visitPermittedSubclass, visitInnerClass, visitField, visitRecordComponent and, of course,
         * visitMethod (see {@link ClassVisitor} javadoc).
         * Functional Utility: Ensures that the processed class is marked as instrumented, preventing subsequent
         *                     (and potentially redundant) instrumentation passes for the same class.
         */
        private void addClassAnnotationIfNeeded() {
            /**
             * Block Logic: Checks if the {@link EntitlementInstrumented} annotation is flagged as needing to be added.
             * Functional Utility: If the flag is set, the annotation is added to the class, and the flag is reset
             *                     to prevent further additions.
             */
            if (annotationNeeded) {
                // logger.debug("Adding {} annotation", ENTITLEMENT_ANNOTATION);
                AnnotationVisitor av = cv.visitAnnotation(ENTITLEMENT_ANNOTATION_DESCRIPTOR, true);
                if (av != null) {
                    av.visitEnd();
                }
                annotationNeeded = false;
            }
        }
    }

    /**
     * @brief A custom {@link MethodVisitor} responsible for injecting entitlement check bytecode into methods.
     * Functional Utility: This visitor is activated for methods identified as requiring an entitlement check.
     *                     It inserts bytecode at the beginning of the method to:
     *                     1. Retrieve the entitlement checker instance.
     *                     2. Determine the caller class.
     *                     3. Forward the original method's arguments.
     *                     4. Invoke the specific entitlement check method.
     *
     * It handles different method types (static, constructor, instance) and adapts to the presence
     * of {@code @CallerSensitive} annotations.
     */
    class EntitlementMethodVisitor extends MethodVisitor {
        /**
         * @brief Flag indicating if the method being instrumented is static.
         * Functional Utility: Influences how method arguments are handled during bytecode injection (e.g., no `this` reference for static methods).
         */
        private final boolean instrumentedMethodIsStatic;
        /**
         * @brief Flag indicating if the method being instrumented is a constructor.
         * Functional Utility: Influences how local variables are indexed and handled in the injected bytecode.
         */
        private final boolean instrumentedMethodIsCtor;
        /**
         * @brief The descriptor of the method being instrumented.
         * Functional Utility: Provides the signature needed to correctly push and pop method arguments.
         */
        private final String instrumentedMethodDescriptor;
        /**
         * @brief The {@link CheckMethod} containing details about the entitlement check to be injected.
         * Functional Utility: Specifies the target checker class, method name, and parameter types for the entitlement invocation.
         */
        private final CheckMethod checkMethod;
        /**
         * @brief Flag indicating if the method has a {@code @CallerSensitive} annotation.
         * Functional Utility: Determines which reflection mechanism to use for obtaining the caller class.
         */
        private boolean hasCallerSensitiveAnnotation = false;

        /**
         * @brief Constructs a new EntitlementMethodVisitor.
         * @param api The ASM API version being used.
         * @param methodVisitor The next {@link MethodVisitor} in the chain.
         * @param instrumentedMethodIsStatic `true` if the method being instrumented is static.
         * @param instrumentedMethodIsCtor `true` if the method being instrumented is a constructor.
         * @param instrumentedMethodDescriptor The descriptor of the method being instrumented.
         * @param checkMethod The {@link CheckMethod} to use for instrumentation.
         * Functional Utility: Initializes the visitor with contextual information about the method to be instrumented.
         */
        EntitlementMethodVisitor(
            int api,
            MethodVisitor methodVisitor,
            boolean instrumentedMethodIsStatic,
            boolean instrumentedMethodIsCtor,
            String instrumentedMethodDescriptor,
            CheckMethod checkMethod
        ) {
            super(api, methodVisitor);
            this.instrumentedMethodIsStatic = instrumentedMethodIsStatic;
            this.instrumentedMethodIsCtor = instrumentedMethodIsCtor;
            this.instrumentedMethodDescriptor = instrumentedMethodDescriptor;
            this.checkMethod = checkMethod;
        }

        /**
         * @brief Visits an annotation of the method.
         * @param descriptor The class descriptor of the annotation class.
         * @param visible `true` if the annotation is visible at runtime.
         * @return An {@link AnnotationVisitor} to visit the annotation's values.
         * Functional Utility: Detects the presence of {@code @CallerSensitive} annotation on the method,
         *                     which affects how the caller class is determined during instrumentation.
         */
        @Override
        public AnnotationVisitor visitAnnotation(String descriptor, boolean visible) {
            /**
             * Block Logic: Checks if the visited annotation is {@code @CallerSensitive} and visible at runtime.
             * Functional Utility: Sets a flag that influences the bytecode generation for obtaining the caller class.
             */
            if (visible && descriptor.endsWith("CallerSensitive;")) {
                hasCallerSensitiveAnnotation = true;
            }
            return super.visitAnnotation(descriptor, visible);
        }

        /**
         * @brief Visits the start of the method's code.
         * Functional Utility: This is the entry point for injecting the entitlement check bytecode.
         *                     The `pushEntitlementChecker`, `pushCallerClass`, `forwardIncomingArguments`,
         *                     and `invokeInstrumentationMethod` calls are made here to ensure the check
         *                     executes before any of the original method's logic.
         */
        @Override
        public void visitCode() {
            pushEntitlementChecker();
            pushCallerClass();
            forwardIncomingArguments();
            invokeInstrumentationMethod();
            super.visitCode();
        }

        /**
         * @brief Injects bytecode to push the entitlement checker instance onto the stack.
         * Functional Utility: Retrieves the singleton instance of the entitlement checker
         *                     class (e.g., `EntitlementChecker.instance()`) and casts it
         *                     to the expected type.
         */
        private void pushEntitlementChecker() {
            mv.visitMethodInsn(INVOKESTATIC, handleClass, "instance", getCheckerClassMethodDescriptor, false);
            mv.visitTypeInsn(CHECKCAST, checkMethod.className());
        }

        /**
         * @brief Injects bytecode to push the caller class onto the stack.
         * Functional Utility: Dynamically determines and provides the {@link Class} object
         *                     of the method that invoked the currently instrumented method.
         *                     It adapts its approach based on the presence of the {@code @CallerSensitive}
         *                     annotation for optimal performance or compatibility.
         */
        private void pushCallerClass() {
            /**
             * Block Logic: Decides whether to use `jdk.internal.reflect.Reflection.getCallerClass` or a custom utility method.
             * Functional Utility: If {@code @CallerSensitive} is present, it uses the optimized JDK internal reflection;
             *                     otherwise, it falls back to a custom utility, balancing performance and broad applicability.
             */
            if (hasCallerSensitiveAnnotation) {
                mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/internal/reflect/Reflection",
                    "getCallerClass",
                    Type.getMethodDescriptor(Type.getType(Class.class)),
                    false
                );
            } else {
                mv.visitMethodInsn(
                    INVOKESTATIC,
                    "org/elasticsearch/entitlement/bridge/Util",
                    "getCallerClass",
                    Type.getMethodDescriptor(Type.getType(Class.class)),
                    false
                );
            }
        }

        /**
         * @brief Injects bytecode to forward the original method's arguments onto the stack.
         * Functional Utility: Prepares the method's arguments to be passed to the injected
         *                     entitlement check method, ensuring the check has access to the
         *                     context of the original method call.
         */
        private void forwardIncomingArguments() {
            int localVarIndex = 0;
            /**
             * Block Logic: Adjusts the local variable index based on whether the method is a constructor or an instance method.
             * Functional Utility: Skips the implicit `this` reference for constructors (which is at index 0) and
             *                     accounts for it for regular instance methods, ensuring correct argument loading.
             */
            if (instrumentedMethodIsCtor) {
                localVarIndex++; // 'this' is at index 0
            } else if (instrumentedMethodIsStatic == false) {
                mv.visitVarInsn(Opcodes.ALOAD, localVarIndex++); // load 'this' at index 0 for instance methods
            }
            /**
             * Block Logic: Iterates through the argument types of the instrumented method.
             * Functional Utility: Loads each argument from its local variable slot onto the operand stack,
             *                     making them available for the subsequent entitlement check method invocation.
             */
            for (Type type : Type.getArgumentTypes(instrumentedMethodDescriptor)) {
                mv.visitVarInsn(type.getOpcode(Opcodes.ILOAD), localVarIndex);
                localVarIndex += type.getSize();
            }
        }

        /**
         * @brief Injects bytecode to invoke the specified entitlement check method.
         * Functional Utility: Calls the actual entitlement logic with the prepared arguments (checker instance,
         *                     caller class, original method arguments), triggering the access control.
         */
        private void invokeInstrumentationMethod() {
            mv.visitMethodInsn(
                INVOKEINTERFACE,
                checkMethod.className(),
                checkMethod.methodName(),
                Type.getMethodDescriptor(
                    Type.VOID_TYPE,
                    checkMethod.parameterDescriptors().stream().map(Type::getType).toArray(Type[]::new)
                ),
                true
            );
        }

    }

    /**
     * @brief A record to encapsulate information about a class file.
     * @param fileName The name of the class file (e.g., "MyClass.class").
     * @param bytecodes The raw bytecode content of the class file.
     * Functional Utility: Provides a structured way to hold both the identifier and the binary data of a class.
     */
    record ClassFileInfo(String fileName, byte[] bytecodes) {}
}
