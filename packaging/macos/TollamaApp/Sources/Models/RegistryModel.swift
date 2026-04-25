import Foundation

indirect enum JSONValue: Codable, Equatable, CustomStringConvertible, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .null
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    var intValue: Int? {
        switch self {
        case .int(let value):
            return value
        case .double(let value):
            return Int(value)
        case .string(let value):
            return Int(value)
        default:
            return nil
        }
    }

    var stringValue: String? {
        if case .string(let value) = self {
            return value
        }
        return nil
    }

    var description: String {
        switch self {
        case .string(let value):
            return value
        case .int(let value):
            return String(value)
        case .double(let value):
            return String(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .object(let value):
            return "{\(value.keys.sorted().joined(separator: ", "))}"
        case .array(let value):
            return "\(value.count) items"
        case .null:
            return "null"
        }
    }
}

struct V1ModelsResponse: Decodable, Sendable {
    let available: [RegistryModel]
    let installed: [RegistryModel]
}

struct RegistryModel: Decodable, Identifiable, Equatable, Sendable {
    let name: String
    let family: String
    let installed: Bool
    let license: RegistryLicense
    let source: ModelSourceInfo?
    let metadata: [String: JSONValue]
    let capabilities: ModelCapabilitiesInfo?

    var id: String { name }

    var maxHorizon: Int? {
        metadata["max_horizon"]?.intValue ?? metadata["prediction_length"]?.intValue
    }

    var implementation: String? {
        metadata["implementation"]?.stringValue
    }

    var isDemoModel: Bool {
        name == "mock" || family == "mock" || source?.type == "local"
    }

    enum CodingKeys: String, CodingKey {
        case name
        case family
        case installed
        case license
        case source
        case metadata
        case capabilities
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        family = try container.decode(String.self, forKey: .family)
        installed = try container.decode(Bool.self, forKey: .installed)
        license = try container.decode(RegistryLicense.self, forKey: .license)
        source = try container.decodeIfPresent(ModelSourceInfo.self, forKey: .source)
        metadata = try container.decodeIfPresent([String: JSONValue].self, forKey: .metadata) ?? [:]
        capabilities = try container.decodeIfPresent(ModelCapabilitiesInfo.self, forKey: .capabilities)
    }
}

struct RegistryLicense: Decodable, Equatable, Sendable {
    let type: String?
    let needsAcceptance: Bool
    let accepted: Bool?
    let notice: String?

    enum CodingKeys: String, CodingKey {
        case type
        case needsAcceptance = "needs_acceptance"
        case accepted
        case notice
    }
}

struct ModelSourceInfo: Decodable, Equatable, Sendable {
    let type: String
    let repoID: String
    let revision: String

    enum CodingKeys: String, CodingKey {
        case type
        case repoID = "repo_id"
        case revision
    }
}

struct ModelCapabilitiesInfo: Decodable, Equatable, Sendable {
    let pastCovariatesNumeric: Bool
    let pastCovariatesCategorical: Bool
    let futureCovariatesNumeric: Bool
    let futureCovariatesCategorical: Bool
    let staticCovariates: Bool

    enum CodingKeys: String, CodingKey {
        case pastCovariatesNumeric = "past_covariates_numeric"
        case pastCovariatesCategorical = "past_covariates_categorical"
        case futureCovariatesNumeric = "future_covariates_numeric"
        case futureCovariatesCategorical = "future_covariates_categorical"
        case staticCovariates = "static_covariates"
    }

    var summary: String {
        var parts: [String] = []
        if pastCovariatesNumeric {
            parts.append("past numeric")
        }
        if pastCovariatesCategorical {
            parts.append("past categorical")
        }
        if futureCovariatesNumeric {
            parts.append("future numeric")
        }
        if futureCovariatesCategorical {
            parts.append("future categorical")
        }
        if staticCovariates {
            parts.append("static")
        }
        return parts.isEmpty ? "target only" : parts.joined(separator: ", ")
    }
}
