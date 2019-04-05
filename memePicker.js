var child_process = require('child_process');

var pendingMemes = [];
var counter = 0;
var totWaitTime = 0;
var totServiceTime = 0;
var counterAlert = 4;
var noticeLength = 2;

function resetTimings() {
    counter = 0;
    totWaitTime = 0;
    totServiceTime = 0;
    counterAlert = 16;
    noticeLength = 2;
}

var memePicker = child_process.spawn("java", ["-cp", "DocClustering.jar", "clustering.DocClustering"],
{
    cwd: __dirname
});

exports.verbose = true;
/*
memePicker.stdout.on("data", function (data) {
    var meme = data.toString().trim();  
 
    if (meme == "") {
        console.log("Got empty line from MemeClassifier.jar");
        return;
    }
    if (pendingMemes.length == 0) {
        console.error("Got unexpected from MemeClassifier.jar: " + data.toString().charCodeAt(1));
        console.log(typeof data);
        return;
    }
    var entry = pendingMemes.shift();
    counter++;
    if (counter == counterAlert && exports.verbose) {
        console.log(exports.timings());
        counterAlert *= 2;
    }
    totWaitTime += (entry.serviceTime - entry.entryTime);
    totServiceTime += ((new Date()) - entry.serviceTime);
    console.log(pendingMemes.length);
  //  if (pendingMemes.length > 0) startNextMeme();
    entry.callback(meme);
}); */
function startNextMeme() {
    console.assert(pendingMemes.length > 0, "pendingMemes.length > 0");
    var entry = pendingMemes[0];
    console.assert(!entry.serviceTime, "!entry.serviceTime");
    entry.serviceTime = new Date();
    console.log("wrote "+entry.str);
    memePicker.stdin.write(entry.str + "\n");
}

/*
exports.getMeme = function (str, callback) {
    pendingMemes.push({
        str: str.trim().replace(/[\n\r]/g, ""),
        callback: callback,
        entryTime: new Date(),
    });
    if (exports.verbose && pendingMemes.length == noticeLength) {
        console.log(exports.timings());
        noticeLength *= 2;
    }
    console.log("length "+pendingMemes.length);
    if (pendingMemes.length == 1) startNextMeme();
}*/

exports.getMeme = function (str, callback) {
	  memePicker.stdin.write(str + "\n");
	  memePicker.stdout.on("data", function (data) {callback(data);});
}

exports.timings = function () {
    var res = pendingMemes.length + "\tpending memes\n";
    res += counter + "\tmemes serverd\n";
    if (counter > 0) {
        res += totServiceTime / counter + "ms\taverage service time\n";
        res += totWaitTime / counter + "ms\taverage wait time\n";
    }
    return res;
}


memeTests = [["y u no think?", "Y-U-No"], ["You dawg go", "Yo-Dawg"], ["Might I inquire as to the identity of the individual Who has released the canines who were being held in captivity?", "Joseph-Ducreux"]];
/* WEN, for some reason, just the string "YOU DAWG" doesn't give "YO-DAWG" */

exports.testMemes = function(N) {
    N = N || 1000;
    console.log("TESTING " + N + " memes");
    var i;
    for (i = 0; i < N; i++) {
        var test = memeTests[Math.floor(Math.random() * memeTests.length)];
        exports.getMeme(test[0], function (test) {
            return function (meme) {
                console.assert(meme === test[1], "MEME test failure: " + test[0] + ".\t Expected " + test[1] + ", got " + meme);
            }
        }(test));
    }
    setTimeout(function () { console.log(exports.timings()); resetTimings(); }, 10000);
}

//setTimeout(exports.testMemes, 10000); // Wait 10 seconds so the child process can start.