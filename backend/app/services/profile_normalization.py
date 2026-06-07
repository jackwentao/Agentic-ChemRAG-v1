from backend.app.domain.documents import ProfileEntity


class ProfileNormalizer:
    def normalize_entities(self, entities: list[str]) -> list[str]:
        return list(dict.fromkeys(entity for entity in entities if entity.strip()))

    def normalize_topics(self, topics: list[str]) -> list[str]:
        return list(dict.fromkeys(topic for topic in topics if topic.strip()))[:12]

    def typed_entities(self, entities: list[str]) -> list[ProfileEntity]:
        entity_types = {
            "porphyrin": "material",
            "phthalocyanine": "material",
            "zinc": "chemical",
            "ZnO": "material",
            "polymer": "material",
            "COF": "material",
            "luminescence": "property",
            "nonlinear optics": "property",
        }
        return [
            ProfileEntity(
                name=entity,
                entity_type=entity_types.get(entity, "topic"),
                normalized_name=entity,
                confidence=1.0,
            )
            for entity in self.normalize_entities(entities)
        ]
