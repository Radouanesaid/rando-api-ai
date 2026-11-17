#!/bin/bash
ask() {
  echo "Q: $1"
  curl -s -X POST http://localhost:8081/ask -H "Content-Type: application/json" -d "{\"question\": \"$1\"}"
  echo -e "\n"
}
ask "une rando facile"
ask "qu'est-ce qui s'est passe a Fontainebleau ?"
ask "depuis quand existe le club ?"
ask "quelle est la meteo demain ?"
