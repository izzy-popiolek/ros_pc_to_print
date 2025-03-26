install(
  DIRECTORY package.xml
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  data/
  DESTINATION share/${PROJECT_NAME}/data
)


