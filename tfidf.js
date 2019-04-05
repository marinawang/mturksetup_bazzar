var natural = require('natural'),
    TfIdf = natural.TfIdf,
  tokenizer = new natural.WordTokenizer();
var tfidf = new TfIdf();  
  
exports.buildcorpus = function () {
	
//	tfidf.addFileSync('data_files/one.txt');
//	tfidf.addFileSync('data_files/two.txt');

tfidf.addDocument(['document', 'about', 'node']);
tfidf.addDocument(['document', 'about', 'ruby']);
tfidf.addDocument(['document', 'about', 'ruby', 'node']);
tfidf.addDocument(['document', 'about', 'node', 'node', 'examples']);

console.log(tfidf.tfidf('node', 0));

console.log('node --------------------------------');
tfidf.tfidfs('node', function(i, measure) {
    console.log('document #' + i + ' is ' + measure);
});

console.log('ruby --------------------------------');
tfidf.tfidfs('ruby', function(i, measure) {
    console.log('document #' + i + ' is ' + measure);
});

}


exports.documentvector = function(text) {
	
	var words = tokenizer.tokenize(text);
	console.log(words);
//	for(var k=0;k<words.length;k++ ){
		//words[k]
	tfidf.tfidfs('node', function(i, measure) {
    console.log('document #' + i + ' is ' + measure);    
  });
//	}
}

 