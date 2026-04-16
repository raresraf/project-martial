/**
 * @file ElasticTextEmbeddingPayload.java
 * @brief Defines the request/response schema for text embedding models hosted on AWS SageMaker,
 *        conforming to a specific Elastic-defined JSON structure. This class acts as a translator
 *        between Elasticsearch's internal representation and the expected SageMaker endpoint format.
 *
 * This payload handler is responsible for:
 * 1.  **Schema Validation**: Ensuring that the user-provided service settings in the model configuration
 *     are valid for text embedding tasks. This includes mandatory fields like `element_type` and `similarity`.
 * 2.  **Request Serialization**: Transforming the incoming Elasticsearch inference request into a
 *     JSON byte stream (`SdkBytes`) that the SageMaker endpoint can understand.
 * 3.  **Response Deserialization**: Parsing the JSON response from the SageMaker endpoint and converting
 *     it into one of the specialized `TextEmbeddingResults` objects (`TextEmbeddingBitResults`,
 *     `TextEmbeddingByteResults`, or `TextEmbeddingFloatResults`) based on the `element_type`
 *     defined in the model's service settings.
 *
 * The class supports multiple embedding vector data types (`bit`, `byte`, `float`), which is a key
 * differentiator from a generic payload. This requires type-specific parsing logic to handle the
 * nuances of each format.
 */
package org.elasticsearch.xpack.inference.services.sagemaker.schema.elastic;

import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParserConfiguration;
import org.elasticsearch.xpack.core.inference.results.TextEmbeddingBitResults;
import org.elasticsearch.xpack.core.inference.results.TextEmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.TextEmbeddingFloatResults;
import org.elasticsearch.xpack.core.inference.results.TextEmbeddingResults;
import org.elasticsearch.xpack.inference.services.sagemaker.SageMakerInferenceRequest;
import org.elasticsearch.xpack.inference.services.sagemaker.model.SageMakerModel;
import org.elasticsearch.xpack.inference.services.sagemaker.schema.SageMakerStoredServiceSchema;

import java.io.IOException;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xcontent.json.JsonXContent.jsonXContent;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalBoolean;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalPositiveInteger;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractRequiredEnum;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractSimilarity;

/**
 * TextEmbedding needs to differentiate between Bit, Byte, and Float types. Users must specify the
 * {@link org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.ElementType} in the Service Settings,
 * and Elastic will use that to parse the request/response. {@link SimilarityMeasure} and Dimensions are also needed, though Dimensions can
 * be guessed and set during the validation call.
 * At the very least, Service Settings must look like:
 * {
 *     "element_type": "bit|byte|float",
 *     "similarity": "cosine|dot_product|l2_norm"
 * }
 */
public class ElasticTextEmbeddingPayload implements ElasticPayload {
    private static final EnumSet<TaskType> SUPPORTED_TASKS = EnumSet.of(TaskType.TEXT_EMBEDDING);
    private static final ParseField EMBEDDING = new ParseField("embedding");

    /**
     * Specifies the machine learning task types that this payload schema supports.
     *
     * @return An `EnumSet` containing only `TaskType.TEXT_EMBEDDING`, indicating that this schema is
     *         exclusively for text embedding inference operations.
     */
    @Override
    public EnumSet<TaskType> supportedTasks() {
        return SUPPORTED_TASKS;
    }

    /**
     * Parses and validates the service-specific settings from the model configuration map.
     *
     * @param serviceSettings A map containing the API-specific configurations provided by the user.
     * @param validationException An exception object to which validation errors are added.
     * @return An `ApiServiceSettings` record containing the validated and structured settings.
     *         This method extracts critical parameters like `element_type`, `similarity`, and `dimensions`,
     *         which are essential for correct request/response handling.
     */
    @Override
    public SageMakerStoredServiceSchema apiServiceSettings(Map<String, Object> serviceSettings, ValidationException validationException) {
        return ApiServiceSettings.fromMap(serviceSettings, validationException);
    }

    /**
     * Serializes the inference request into an `SdkBytes` object, which represents the HTTP request body
     * sent to the SageMaker endpoint.
     *
     * @param model The `SageMakerModel` containing configuration details, including the validated API service settings.
     * @param request The `SageMakerInferenceRequest` containing the input data to be processed.
     * @return An `SdkBytes` object containing the JSON payload for the SageMaker endpoint.
     * @throws Exception if the service settings are of an unexpected type, indicating a schema mismatch.
     */
    @Override
    public SdkBytes requestBytes(SageMakerModel model, SageMakerInferenceRequest request) throws Exception {
        if (model.apiServiceSettings() instanceof ApiServiceSettings) {
            return ElasticPayload.super.requestBytes(model, request);
        } else {
            throw createUnsupportedSchemaException(model);
        }
    }

    /**
     * Registers the custom `ApiServiceSettings` class as a named writeable object, allowing it to be
     * serialized and deserialized across the Elasticsearch cluster.
     *
     * @return A `Stream` of `NamedWriteableRegistry.Entry` objects, including the one for `ApiServiceSettings`.
     *         This is critical for ensuring that model configurations can be persistently stored and transmitted
     *         between nodes.
     */
    @Override
    public Stream<NamedWriteableRegistry.Entry> namedWriteables() {
        return Stream.concat(
            ElasticPayload.super.namedWriteables(),
            Stream.of(
                new NamedWriteableRegistry.Entry(SageMakerStoredServiceSchema.class, ApiServiceSettings.NAME, ApiServiceSettings::new)
            )
        );
    }

    /**
     * Deserializes the `InvokeEndpointResponse` from SageMaker into a structured `TextEmbeddingResults` object.
     *
     * @param model The `SageMakerModel` used for the inference, which contains the `elementType` setting.
     * @param response The raw response from the SageMaker `invoke_endpoint` API call.
     * @return A `TextEmbeddingResults` object specialized for the data type (`bit`, `byte`, or `float`)
     *         specified in the model's configuration.
     * @throws Exception if parsing the response body fails.
     *
     * @implNote This method uses a `switch` statement on the `elementType` to delegate parsing to the
     *           appropriate inner class (`TextEmbeddingBinary`, `TextEmbeddingBytes`, or `TextEmbeddingFloat`),
     *           each handling a specific JSON structure for the embedding data.
     */
    @Override
    public TextEmbeddingResults<?> responseBody(SageMakerModel model, InvokeEndpointResponse response) throws Exception {
        try (var p = jsonXContent.createParser(XContentParserConfiguration.EMPTY, response.body().asInputStream())) {
            return switch (model.apiServiceSettings().elementType()) {
                case BIT -> TextEmbeddingBinary.PARSER.apply(p, null);
                case BYTE -> TextEmbeddingBytes.PARSER.apply(p, null);
                case FLOAT -> TextEmbeddingFloat.PARSER.apply(p, null);
            };
        }
    }

    /**
     * Inner class responsible for parsing a JSON response containing binary (bit) embeddings.
     * The expected format is a JSON object with a `text_embedding_bits` field.
     * Reads binary format (it says bytes, but the lengths are different)
     * {
     *     "text_embedding_bits": [
     *         {
     *             "embedding": [
     *                 23
     *             ]
     *         },
     *         {
     *             "embedding": [
     *                 -23
     *             ]
     *         }
     *     ]
     * }
     */
    private static class TextEmbeddingBinary {
        private static final ParseField TEXT_EMBEDDING_BITS = new ParseField(TextEmbeddingBitResults.TEXT_EMBEDDING_BITS);
        @SuppressWarnings("unchecked")
        private static final ConstructingObjectParser<TextEmbeddingBitResults, Void> PARSER = new ConstructingObjectParser<>(
            TextEmbeddingBitResults.class.getSimpleName(),
            IGNORE_UNKNOWN_FIELDS,
            args -> new TextEmbeddingBitResults((List<TextEmbeddingByteResults.Embedding>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), TextEmbeddingBytes.BYTE_PARSER::apply, TEXT_EMBEDDING_BITS);
        }
    }

    /**
     * Inner class for parsing JSON responses with byte-level text embeddings.
     * It expects a `text_embedding_bytes` field and includes range validation to ensure
     * that the numeric values fit within a standard Java `byte`.
     * Reads byte format from
     * {
     *     "text_embedding_bytes": [
     *         {
     *             "embedding": [
     *                 23
     *             ]
     *         },
     *         {
     *             "embedding": [
     *                 -23
     *             ]
     *         }
     *     ]
     * }
     */
    private static class TextEmbeddingBytes {
        private static final ParseField TEXT_EMBEDDING_BYTES = new ParseField("text_embedding_bytes");
        @SuppressWarnings("unchecked")
        private static final ConstructingObjectParser<TextEmbeddingByteResults, Void> PARSER = new ConstructingObjectParser<>(
            TextEmbeddingByteResults.class.getSimpleName(),
            IGNORE_UNKNOWN_FIELDS,
            args -> new TextEmbeddingByteResults((List<TextEmbeddingByteResults.Embedding>) args[0])
        );

        @SuppressWarnings("unchecked")
        private static final ConstructingObjectParser<TextEmbeddingByteResults.Embedding, Void> BYTE_PARSER =
            new ConstructingObjectParser<>(
                TextEmbeddingByteResults.Embedding.class.getSimpleName(),
                IGNORE_UNKNOWN_FIELDS,
                args -> TextEmbeddingByteResults.Embedding.of((List<Byte>) args[0])
            );

        static {
            BYTE_PARSER.declareObjectArray(constructorArg(), (p, c) -> {
                var byteVal = p.shortValue();
                if (byteVal < Byte.MIN_VALUE || byteVal > Byte.MAX_VALUE) {
                    throw new IllegalArgumentException("Value [" + byteVal + "] is out of range for a byte");
                }
                return (byte) byteVal;
            }, EMBEDDING);
            PARSER.declareObjectArray(constructorArg(), BYTE_PARSER::apply, TEXT_EMBEDDING_BYTES);
        }
    }

    /**
     * Inner class dedicated to parsing JSON responses containing floating-point text embeddings.
     * It is designed to handle a `text_embedding` field with an array of floating-point numbers.
     * Reads float format from
     * {
     *     "text_embedding": [
     *         {
     *             "embedding": [
     *                 0.1
     *             ]
     *         },
     *         {
     *             "embedding": [
     *                 0.2
     *             ]
     *         }
     *     ]
     * }
     */
    private static class TextEmbeddingFloat {
        private static final ParseField TEXT_EMBEDDING_FLOAT = new ParseField("text_embedding");
        @SuppressWarnings("unchecked")
        private static final ConstructingObjectParser<TextEmbeddingFloatResults, Void> PARSER = new ConstructingObjectParser<>(
            TextEmbeddingByteResults.class.getSimpleName(),
            IGNORE_UNKNOWN_FIELDS,
            args -> new TextEmbeddingFloatResults((List<TextEmbeddingFloatResults.Embedding>) args[0])
        );

        @SuppressWarnings("unchecked")
        private static final ConstructingObjectParser<TextEmbeddingFloatResults.Embedding, Void> FLOAT_PARSER =
            new ConstructingObjectParser<>(
                TextEmbeddingFloatResults.Embedding.class.getSimpleName(),
                IGNORE_UNKNOWN_FIELDS,
                args -> TextEmbeddingFloatResults.Embedding.of((List<Float>) args[0])
            );

        static {
            FLOAT_PARSER.declareFloatArray(constructorArg(), EMBEDDING);
            PARSER.declareObjectArray(constructorArg(), FLOAT_PARSER::apply, TEXT_EMBEDDING_FLOAT);
        }
    }

    /**
     * A record that encapsulates the validated and structured service settings for the text embedding schema.
     * It is a `SageMakerStoredServiceSchema`, making it a serializable part of the model definition.
     *
     * @param dimensions The dimensionality of the embedding vectors. Can be null if not specified by the user.
     * @param dimensionsSetByUser A flag indicating whether the `dimensions` field was explicitly set by the user.
     * @param similarity The similarity measure to be used with the embedding vectors (e.g., cosine, dot_product).
     * @param elementType The data type of the vector elements (`bit`, `byte`, or `float`). This is a mandatory field
     *                    used to determine how to parse the SageMaker response.
     * Element Type is required. It is used to disambiguate between binary embeddings and byte embeddings.
     */
    record ApiServiceSettings(
        @Nullable Integer dimensions,
        Boolean dimensionsSetByUser,
        @Nullable SimilarityMeasure similarity,
        DenseVectorFieldMapper.ElementType elementType
    ) implements SageMakerStoredServiceSchema {

        private static final String NAME = "sagemaker_elastic_text_embeddings_service_settings";
        private static final String DIMENSIONS_FIELD = "dimensions";
        private static final String DIMENSIONS_SET_BY_USER_FIELD = "dimensions_set_by_user";
        private static final String SIMILARITY_FIELD = "similarity";
        private static final String ELEMENT_TYPE_FIELD = "element_type";

        ApiServiceSettings(StreamInput in) throws IOException {
            this(
                in.readOptionalVInt(),
                in.readBoolean(),
                in.readOptionalEnum(SimilarityMeasure.class),
                in.readEnum(DenseVectorFieldMapper.ElementType.class)
            );
        }

        @Override
        public String getWriteableName() {
            return NAME;
        }

        @Override
        public TransportVersion getMinimalSupportedVersion() {
            return TransportVersions.ML_INFERENCE_SAGEMAKER_ELASTIC;
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeOptionalVInt(dimensions);
            out.writeBoolean(dimensionsSetByUser);
            out.writeOptionalEnum(similarity);
            out.writeEnum(elementType);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            if (dimensions != null) {
                builder.field(DIMENSIONS_FIELD, dimensions);
            }
            builder.field(DIMENSIONS_SET_BY_USER_FIELD, dimensionsSetByUser);
            if (similarity != null) {
                builder.field(SIMILARITY_FIELD, similarity);
            }
            builder.field(ELEMENT_TYPE_FIELD, elementType);
            return builder;
        }

        @Override
        public ApiServiceSettings updateModelWithEmbeddingDetails(Integer dimensions) {
            return new ApiServiceSettings(dimensions, false, similarity, elementType);
        }

        static ApiServiceSettings fromMap(Map<String, Object> serviceSettings, ValidationException validationException) {
            var dimensions = extractOptionalPositiveInteger(
                serviceSettings,
                DIMENSIONS_FIELD,
                ModelConfigurations.SERVICE_SETTINGS,
                validationException
            );
            var dimensionsSetByUser = extractOptionalBoolean(serviceSettings, DIMENSIONS_SET_BY_USER_FIELD, validationException);
            var similarity = extractSimilarity(serviceSettings, ModelConfigurations.SERVICE_SETTINGS, validationException);
            var elementType = extractRequiredEnum(
                serviceSettings,
                ELEMENT_TYPE_FIELD,
                ModelConfigurations.SERVICE_SETTINGS,
                DenseVectorFieldMapper.ElementType::fromString,
                EnumSet.allOf(DenseVectorFieldMapper.ElementType.class),
                validationException
            );
            return new ApiServiceSettings(dimensions, dimensionsSetByUser != null && dimensionsSetByUser, similarity, elementType);
        }
    }
}
