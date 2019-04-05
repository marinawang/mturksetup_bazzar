var Changeset = require('../etherpad-lite-win/node_modules/ep_etherpad-lite/static/js/Changeset.js');
var XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;
var dirty = require('dirty');
var db = dirty('../etherpad-lite-win/var/dirty.db');
var fs = require('fs');
var userrevcnt = {};
var BreakException= {};

var cs = "Z:z>1|2=m=b*0|1+1$\n";

var unpacked = Changeset.unpack(cs);

console.log(unpacked);

var preurl = 'http://luoyang.lti.cs.cmu.edu:9001/api/1.2.7/'
var posturl = '?apikey=87b40a9c3818d6cde3d9960db9c4d1a57199ec86fc165f082fbeac072154d559&padID=idea2';

var cururl = preurl+'listAuthorsOfPad'+posturl;

var xmlHttp = null;

xmlHttp = new XMLHttpRequest();
xmlHttp.open("GET", cururl, false);
xmlHttp.send(null);
var authors;   
if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
    //    var res = JSON.parse(xmlHttp.responseText).response;
       authors = JSON.parse(xmlHttp.responseText).data.authorIDs;
     //   console.log(JSON.parse(xmlHttp.responseText).data.authorIDs);
       
} else {
        console.log("Simsimi is down");
}

var chaturl = preurl+'padUsers'+posturl;

xmlHttp.open("GET", chaturl, false);
xmlHttp.send(null);
var users;   
if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
    //    var res = JSON.parse(xmlHttp.responseText).response;
       users = JSON.parse(xmlHttp.responseText).data.padUsers;
       console.log(users);
       
} else {
        console.log("Simsimi is down");
}
    
try {	
//{"key":"pad:idea2:revs:197","val":{"changeset":"Z:rv>2|l=rl=8*3+2$re","meta":{"author":"a.qmSiE6WMCYB4n6Qt","timestamp":1426459296358}}}
var cntuseractivity = 0;	
db.on('load', function() {
	var cnt = 1;
    db.forEach(function(key, val) {
    	
   
    	cnt++;
    	if(cnt > 100)
    	{
    //		throw BreakException;
    	}
//      console.log('Found key: %s, val: %j', key, val);
 //     console.log(cnt);
   //    	var result = JSON.parse(val);
   // 	var author = result.meta.author;
   	
   		if (!(typeof val.meta === "undefined")) {
    	 
    	 	if (!(typeof val.meta.author === "undefined")) {
    	 		
    	 		var user = val.meta.author;
    	// 			console.log(user);
    	 			
    	 			if(authors.indexOf(user)>-1)
    	 			{
    	 				cntuseractivity++;
    	 	//			 console.log(cntuseractivity);
    	 				 
    	 				 if(user in userrevcnt)
    	 				 {
    	 				 	userrevcnt[user]++;
    	 				  }else{
    	 				  	userrevcnt[user] = 1;
    	 				  }
    	 			}
    	 	}
			}
      
    });  
    
    	console.log('All records are saved on disk now here.');
    		 fs.writeFile(__dirname + "/logs/database.json", JSON.stringify(userrevcnt), function (err) { if (err) { console.log("Error writing logs"); console.log(err); } });
    
 });  
//   db.on('read_close', function() {
//     	 fs.writeFile(__dirname + "/logs/database.json", JSON.stringify(userrevcnt), function (err) { if (err) { console.log("Error writing logs"); console.log(err); } });
//    		console.log('All records are saved on disk now.');
//  		});
 
} catch(e) {
    if (e!==BreakException) throw e;
}

 

