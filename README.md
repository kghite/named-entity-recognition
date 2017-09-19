# named-entity-recognition
Independent Study in NER using Hidden Markov Models, Neural Nets, and Convolutional Neural Nets

## Setup

To start, you need to download the Google News word embeddings. They can be found here: https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download

Once the file is downloaded, it needs to be unzipped into the root directory of the project

## Dataest
We are using the CoNLL-2003.

There are 6 tags.
`PER`: Person
`ORG`: Organization
`LOC`: Location
`MISC`: Miscellaneous (this represents things like products and nationalities)
`O`: Not an entity
`<START>`: Start tag
