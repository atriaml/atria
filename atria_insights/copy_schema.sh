ATRIA_BUILD_REGISTRY=true python src/atria_insights/build_registry.py $@
cp src/atria_insights/schema.db ../../atriax/static/atria_insghts/schema.db
