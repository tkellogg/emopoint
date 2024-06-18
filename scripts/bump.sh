#!/bin/zsh
set -euo pipefail
VERSION=$1

if [[ $VERSION == "--publish" ]]
then
    VERSION=$(cat js/package.json | jq -rc '.version')

    echo 'Git/Go'
    git tag v$VERSION
    git push origin v$VERSION

    echo 'JavaScript'
    (cd js && npm publish)
    echo 'Python'
    poetry publish --build <(yes yes | head -n1)
    exit 0
fi

echo '############################'
echo '### JavaScript'
echo '############################'
JS=js/package.json
jq -M '.version = "'$VERSION'"' $JS > $JS.tmp
mv $JS.tmp $JS
echo 'done.'

echo '############################'
echo '### JavaScript'
echo '############################'
PY=pyproject.toml
cat $PY | awk '{ if ($1 == "version") print "version = \"" "'$VERSION'" "\""; else print $0 }' > $PY.tmp
mv $PY.tmp $PY
echo 'done.'