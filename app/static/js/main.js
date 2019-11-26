const NUMBER_OF_ADJECTIVE = 1;
const BLUE_IMAGE_NUMBER = 6;
const RED_IMAGE_NUMBER = 6;
const NEUTRAL_IMAGE_NUMBER = 2;
const BATCH_NUMBER = 12;
const IMAGE_PATH ='static/image/labeledEx/'
//const IMAGE_PATH = 'static/image/21000_25000selected/'
// const IMAGE_PATH = 'static/image/FFHQ_SAMPLE2/FFHQ_SAMPLE2/'
const SAMPLING_MODE = "RANDOM";

/* Tool 기능 관련 변수들 */

let itemsContainer = document.getElementById("items-container");
let todosContainer = document.getElementById("todos-container");
let temp = document.getElementsByClassName("img_temp")[0];
let confirm_button = document.getElementsByClassName("button-confirm")[0];
confirm_button.disabled = false;

let mouseOffset = {x:0, y:0};
let isMouseDown = false;
let currentTodo = null;
let currentList = [];
let tempTodo = null;
let tempTodo_list = [];
let ctrlPressed = false;
let multiChoice = false;
let totalDisplay = null;

let beforeLabel = [];

/* 드래그앤 드랍 */

// let doElsCollide = function(el1, el2) { 
//   if(el1 != null && el2 != null){
    
//     rect1 = new cumulativeOffset(el1);
//     rect2 = new cumulativeOffset(el2);

//     el1.offsetBottom= rect1.top + el1.offsetHeight;
//     el1.offsetRight = rect1.left + el1.offsetWidth;
//     el2.offsetBottom = rect2.top + 15 + el2.offsetHeight;
//     el2.offsetRight = rect2.left - 14 + el2.offsetWidth;

//     return !((el1.offsetBottom < rect2.top) ||
//               (rect1.top > el2.offsetBottom) ||
//               (el1.offsetRight < rect2.left) ||
//               (rect1.left > el2.offsetRight))
//             }

// };

// function onMouseDown_clone(e, item) {
  
//   let isOver = tempTodo.className.includes("over");
//   if(ctrlPressed){
//     if(isOver){
//       tempTodo.className = tempTodo.className.replace(" over","");
//       item.className = tempTodo.className.replace(" over","");

//     }
//     else{
//       multiChoice = true;
//       tempTodo.className +=' over';
//       let multi_list = document.getElementsByClassName('over');   
//     }
//   }
//   else{
//     if(isOver){
//   e.preventDefault();  
//   isMouseDown = true;
//   currentTodo = item;
//   currentList = [];
//   let multi_items = document.getElementsByClassName('over');
//   tempTodo_list = [];
//   for(let i=0;i<multi_items.length;i++){
//     tempTodo_list.push(multi_items[i]);
    
//     let multi_clone = cloneImage(multi_items[i]);
//     if(multi_clone.getAttribute('slot') != item.getAttribute('slot')){
//       $(".img_temp").append(multi_clone);
//       currentList.push(multi_clone)
//     }
//     else{
//       currentList.push(item);
//     }
//   }

//   for(let i=0;i<currentList.length;i++){
//     currentList[i].style.zIndex = "2";  
//     currentList[i].style.filter = "brightness(50%)";
//   }
//   mouseOffset_list = [];
//   for(let i=0;i<currentList.length;i++){
//     mouseOffset_list.push({x: currentList[i].offsetLeft - e.clientX, y: currentList[i].offsetTop - e.clientY});
//     tempTodo_list[i].remove();
//   }
// }
//     else{
//   e.preventDefault();
//   isMouseDown = true;
//   item.style.zIndex = "2";
//   currentList = [];
//   currentList.push(item);
//   currentTodo = item;
//   mouseOffset_list = [];
//   mouseOffset_list.push({x: item.offsetLeft - e.clientX, y: item.offsetTop - e.clientY});
//   tempTodo.remove();
//   item.style.filter = "brightness(50%)";
//     }
//   }
// }

// function onMouseMove_clone(e) {
//   e.preventDefault();
//   if(isMouseDown) {
//     for(let i=0;i<currentList.length;i++){
//       currentList[i].style.left = e.clientX + mouseOffset_list[i].x + "px";
//       currentList[i].style.top = e.clientY + mouseOffset_list[i].y + "px";
//     }
//   }
// }

// function onMouseUp_clone(e, item) {
//   if(ctrlPressed){

//   }
//   else{
//     currentTodo.style.zIndex = "1";
//     isMouseDown = false;
//     item.style.filter = "brightness(100%)";
//   }
// }

// function onMouseOver(e, item) {
//   if(!isMouseDown){

//     item.style.filter = "brightness(130%)";
//     todo_clone = cloneImage(item);
//     tempTodo = item;
    
//     if ( temp.hasChildNodes() ) { temp.removeChild( temp.firstChild ); }
//     $(".img_temp").append(todo_clone);
//     setListener_clone(todo_clone);
//   }
// } 

// var cumulativeOffset = function(element) {
//   var top = 0, left = 0;
//   do {
//       top += element.offsetTop  || 0;
//       left += element.offsetLeft || 0;
//       element = element.offsetParent;
//   } while(element);
//   return {
//       top: top,
//       left: left
//   };
// };

// function cloneImage(item){
//   todo_clone = item.cloneNode();
//     todo_clone.position = "absolute";
//     todo_clone.className = todo_clone.className.replace(" over","");

//     var top = document.body.scrollTop;

//     rect = new cumulativeOffset(item);
//     todo_clone.style.left = rect.left - 14 + "px";
//     todo_clone.style.top = rect.top - 15 - top + "px";
    

//     if(item.parentNode.className == "left"){
//       todo_clone.setAttribute('id', 0);
//     }
//     else if(item.parentNode.className == "right"){
//       todo_clone.setAttribute('id', 2);
//     }
//     else{
//       todo_clone.setAttribute('id', 1);
//     }
//     todo_clone.setAttribute('slot', item.parentNode.id);
//     return todo_clone;
// }

// function onMouseOver_clone(e, item) {
//   if(!isMouseDown){
//     item.style.filter = "brightness(130%)";

//     let jObject = new Object();  
//     jObject.image_id = item.src.split(/[/]+/).pop();
  
//     jQuery.ajaxSettings.traditional = true;
    
//     attrParam = JSON.stringify(jObject);

//     $.ajax({
//       url : "/getAttribute",
//       type: 'POST',
//       data: {"jsonData" : attrParam},
//       dataType:'json',
//       success: function(data) {
//         console.log(data['attribute']);
                  
//             if(typeof $(".attr_on").attr("class") != "undefined"){
//               $(".attr_on").attr("class", $(".attr_on").attr("class").replace(" attr_on",""));
//             }
//         for(let i=0;i<data['attribute'].length;i++){
   
//           $('#'.concat(data['attribute'][i])).attr('class', 'node'.concat(" attr_on"));
//           // $('#'.concat(data['attribute'][i])).css('fill', 'red');
//         }
//       },
//       error: function(x, e) {
//           alert("error");
//       }
//   });
  
//   }
  
// }


// function onMouseOut(e, item) {
//   if(!isMouseDown){
//     item.style.filter = "brightness(100%)";
//   }
// }

// function onMouseOut_clone(e, item) {
//   if(!isMouseDown){  
//   if(typeof $(".attr_on").attr("class") != "undefined"){
//     $(".attr_on").attr("class", $(".attr_on").attr("class").replace(" attr_on",""));
//   }
//   item.style.filter = "brightness(100%)";
//     item.remove();
//     if ( temp.hasChildNodes() ) { temp.removeChild( temp.firstChild ); }
//   }
// }

// function setListeners(todoItems) {
//   for(let i = 0; i < todoItems.length; i++) {
//   let item = todoItems[i];
//   item.addEventListener("mouseout", (e) => {onMouseOut(e, item); });
//   item.addEventListener("mouseover", (e) => { onMouseOver(e, item); });
//   }
// }
 
// function setListener(todoItem) {
//   todoItem.addEventListener("mouseout", (e) => {onMouseOut(e, todoItem); });
//   todoItem.addEventListener("mouseover", (e) => { onMouseOver(e, todoItem); });
// }
  
// function setListener_clone(todoItem) {
  
//   todoItem.addEventListener("mousedown", (e) => { onMouseDown_clone(e, todoItem); });
//   todoItem.addEventListener("mouseover", (e) => { onMouseOver_clone(e, todoItem); });
//   todoItem.addEventListener("mouseout", (e) => { onMouseOut_clone(e, todoItem); });
//   document.body.addEventListener("mousemove", (e) => {
//     onMouseMove_clone(e);
//   });
//   todoItem.addEventListener("mouseup", (e) => {
//     onMouseUp_clone(e, todoItem);
//   }); 
// }

// /* 매 0.1초마다 실행하는 함수 */
// setInterval(() => {
//   let areas = document.getElementsByClassName("red-blue");
  
//   /* 매 0.1초마다 emptyCheck 변수로 마지막 row가 비어있으면 image row를 삭제해 동적으로 container의 크기를 결정*/
//   let check = 0;
//   for(let i = 0; i < areas.length; i++) {
//     emptyCheck = 0;
//     let lastRows = areas[i].getElementsByClassName('image_row')[areas[i].getElementsByClassName('image_row').length-1].getElementsByTagName('div');
//     for(let j=0; j < lastRows.length; j++){
//       if(!lastRows[j].hasChildNodes()){
//         emptyCheck = emptyCheck + 1;
//       }
//     }
//     if(
//       emptyCheck == lastRows.length
//       && areas[i].getElementsByClassName('image_row').length > 3){
//       areas[i].getElementsByClassName('image_row')[areas[i].getElementsByClassName('image_row').length-1].remove();
//     }
    

//     //check : 아이템이 속한 container의 수를 체크하는 변수
//     // doElsCollide 함수로 현재 드래그 중인 아이템과 container area가 위치상 겹쳐지게 되면 container area에 빨간 줄을 띄우고
//     // 그 상태에서 마우스 버튼이 올라가게 되면 그 container로 드래그 중인 이미지를 삽입

//     areas[i].className = areas[i].className.replace("cont_on", "");
    
//       if(doElsCollide(currentTodo, areas[i])) {
//         areas[i].className += " cont_on";
//         check = check + 1;
//         if(!isMouseDown) {
//           for(let j=0; j < currentList.length; j++){
//             snapTodo(currentList[j], areas[i], i);
//           }
//           currentList = [];
        
//       }
//     }
//   }  
    

//   //check ==0, 드래그 중인 이미지가 겹쳐지는 container가 없을 때, 마우스 버튼이 올라가게 되면
//   // 해당 이미지의 원래 container로 이미지를 삽입
//     if(check == 0 && currentTodo != null) {
//       if(!isMouseDown) {
//         let i = currentTodo.getAttribute('id');
//         for(let j=0; j < currentList.length; j++){
//           snapTodo(currentList[j], areas[i], i);
//         }
//         currentList = [];
//       }
//     }
  

  
// }, 100);

// //container에 todo이미지를 삽입하는 함수
// function snapTodo(todo, container,index) {
//   area_list = ["left","center","right"];
//   id_list = ["L","N","R"];
//   if(typeof $(".attr_on").attr("class") != "undefined"){
//     $(".attr_on").attr("class", $(".attr_on").attr("class").replace(" attr_on",""));
//   }

// //log data를 저장하기 위한 jObject 선언
//   let jObject = new Object();
//   jObject.Time = js_yyyy_mm_dd_hh_mm_ss ();
//   jObject.adjective = keyword;
//   jObject.What = todo.src.split(/[/]+/).pop();
//   if(todo.getAttribute('id') == 0){
//     jObject.From = "left"
//   }
//   else if(todo.getAttribute('id') == 1){
//     jObject.From = "center"
//   }
//   else{
//     jObject.From = "right"
//   }
//   jObject.To = area_list[index]
//   logParam = JSON.stringify(jObject)

// //ajax 통신을 통해 이미지가 어디서 어디로 옮겨졌는지 데이터베이스에 로그데이터 저장
//   jQuery.ajaxSettings.traditional = true;
//   $.ajax({
//     url : "/getLog",
//     type: 'POST',
//     data: {"jsonData" : logParam},
//     dataType:'json',
//     success: function(data) {
//     },
//     error: function(x, e) {
//         alert("error");
//     }
// });

//   // 드래그 중인 이미지가 image row에 여유 공간이 있으면 해당 image row에 image를 append
//   let fullCount = 0;
//   let lastID = "";
//   let row_count = container.getElementsByClassName('image_row')[0].childElementCount;
//     temp_list = document.getElementsByClassName(area_list[index]);
//     for(let i=0;i<temp_list.length;i++){
//       let item = temp_list[i];
//       fullCount = fullCount + 1;
//       if(i == (temp_list.length - 1)){
//         lastID = item.id;
//       }
//       if(!item.hasChildNodes()){
//         todo_clone = todo.cloneNode();
//         todo.remove();
//         item.append(todo_clone);
//         todo_clone.className = todo_clone.className.replace(" over","");
//         setListener(todo_clone);
//         todo_clone.style.left = 0 + "px";
//         todo_clone.style.top = 0 + "px";
//         todo_clone.style.filter = "brightness(100%)";
//         currentTodo = null;
//         fullCount = 0;
//         break;
//       }
//     }
    
//     // 만약 full count가 전체 row의 개수와 같다(모든 image row가 image로 가득 찼다면) 새로운 image row를 만들어 div 확장
//     if(fullCount == temp_list.length){
//       new_row = document.createElement('div');
//       for(let i=1;i<=row_count;i++){  
//         new_slot = document.createElement('div');
//         new_slot.className = area_list[index];
//         new_slot.id = id_list[index] + (parseInt(lastID.replace(id_list[index],""))+i);
        
//         if(i==1){
//           todo_clone = todo.cloneNode();
//           todo.remove();
//           new_slot.append(todo_clone);
//           todo_clone.className = todo_clone.className.replace(" over","");  
//           setListener(todo_clone);
//           todo_clone.style.left = 0 + "px";
//           todo_clone.style.top = 0 + "px";
//           currentTodo = null;
//           fullCount = 0; 
//         }
//         new_row.append(new_slot); 
//       }
//       new_row.className = "image_row";
//       container.append(new_row);
//     }
//   }
// /*------------------------------------------------------------------*/
// /* 키보드 기능 */

// document.addEventListener("keydown", checkKeyPressed, false);
// document.addEventListener("keyup", checkKeyUp, false);
// document.addEventListener("mousedown", checkMousedown, false);

// //multiChoice 상태에서 외부 지점을 클릭하면 multiChoice된 image 모두 해제
// function checkMousedown(e) {
//   if(multiChoice && !ctrlPressed){
//     let multi_list = document.getElementsByClassName('over');
//     let multi_length = multi_list.length;
//     for(let i=0;i<multi_length;i++){
//       multi_list[0].className = multi_list[0].className.replace(" over",""); 
//     }
//   }
// }

// //키보드 관련 컨트롤 함수

// //control 키 눌림
// function checkKeyPressed(e) {
//   if (e.keyCode == "17" || e.keyCode == "91") {
//       ctrlPressed = true;
//   }
// }

// function checkKeyUp(e) {

// //control 키 떼짐
//   if (e.keyCode == "17"|| e.keyCode == "91") {
//     ctrlPressed = false;
//   }

// //multi_choice된 상태에서 'a' key가 떼지면 모두 왼쪽으로 이동
//   else if (e.keyCode == "65"){
//   if ( temp.hasChildNodes() ) { temp.removeChild( temp.firstChild ); }
//   let multi_list = document.querySelectorAll('.over');
//   let areas = document.getElementsByClassName("red-blue");
  
//   for(let i=0;i<multi_list.length;i++){
//       snapTodo(multi_list.item(i),areas[0],0);
//     }
//   }

// //multi_choice된 상태에서 's' key가 떼지면 모두 가운데쪽으로 이동
//   else if (e.keyCode == "87"){
//     if ( temp.hasChildNodes() ) { temp.removeChild( temp.firstChild ); }
//   let multi_list = document.querySelectorAll('.over');  
//   let areas = document.getElementsByClassName("red-blue");
//     for(let i=0;i<multi_list.length;i++){
//       snapTodo(multi_list.item(i),areas[1],1);
//     }
//   }
  
// //multi_choice된 상태에서 'd' key가 떼지면 모두 오른쪽으로 이동
//   else if (e.keyCode == "68"){
//   if ( temp.hasChildNodes() ) { temp.removeChild( temp.firstChild ); }
//   let multi_list = document.querySelectorAll('.over');
//   let areas = document.getElementsByClassName("red-blue");
//     for(let i=0;i<multi_list.length;i++){
//       snapTodo(multi_list.item(i),areas[2],2);
//     }
//   }

// //'space bar' key가 떼지면 confirm
//   else if (e.keyCode == "83"){
//     if(confirm_button.disabled == false){
//       confirm_click();

//     }
//   }
// }

// /*------------------------------------------------------------------*/


function setListeners(todoItems) {
  for(let i = 0; i < todoItems.length; i++) {
  let item = todoItems[i];
  item.addEventListener("mouseout", (e) => {onMouseOut(e, item); });
  item.addEventListener("mouseover", (e) => { onMouseOver(e, item); });
  item.addEventListener("mousedown", (e) => { onMouseDown(e, item); });
  }
}

function setListener(todoItem) {
  todoItem.addEventListener("mouseout", (e) => {onMouseOut(e, todoItem); });
  todoItem.addEventListener("mouseover", (e) => { onMouseOver(e, todoItem); });
  todoItem.addEventListener("mousedown", (e) => { onMouseDown(e, todoItem); });
}


function onMouseOver  (e, item) {
  if(!isMouseDown){
    item.style.filter = "brightness(130%)";

    let jObject = new Object();  
    jObject.image_id = item.src.split(/[/]+/).pop();
  
    jQuery.ajaxSettings.traditional = true;
    
    attrParam = JSON.stringify(jObject);

    /*$.ajax({
      url : "/getAttribute",
      type: 'POST',
      data: {"jsonData" : attrParam},
      dataType:'json',
      success: function(data) {
        console.log(data['attribute']);
                  
            if(typeof $(".attr_on").attr("class") != "undefined"){
              $(".attr_on").attr("class", $(".attr_on").attr("class").replace(" attr_on",""));
            }
        for(let i=0;i<data['attribute'].length;i++){
   
          $('#'.concat(data['attribute'][i])).attr('class', 'node'.concat(" attr_on"));
          // $('#'.concat(data['attribute'][i])).css('fill', 'red');
        }
      }, 
      error: function(x, e) {
          alert("error");
      }
  }); */ 
}
}

function onMouseOut(e, item) {
  if(!isMouseDown){
    item.style.filter = "brightness(100%)";

  if(typeof $(".attr_on").attr("class") != "undefined"){
    $(".attr_on").attr("class", $(".attr_on").attr("class").replace(" attr_on",""));
  }
  }
}

function onMouseDown(e, item) {
  
  let isOver = item.className.includes("over");

    if(isOver){
      item.className = item.className.replace(" over","");
      let jObject = new Object();  
      jObject.image_id = item.src.split(/[/]+/).pop();
      jObject.From = "positive";
      jObject.To = "negative";
      jObject.batch = count_num;
      jObject.Time = js_yyyy_mm_dd_hh_mm_ss ();
      jObject.adjective = keyword;

      jQuery.ajaxSettings.traditional = true;
      
      logParam = JSON.stringify(jObject);
      
      $.ajax({
        url : "/getLog",
        type: 'POST',
        data: {"jsonData" : logParam},
        dataType:'json',
        success: function(data) {
          
        },
        error: function(x, e) {
            alert("error");
        }
    });


    }
    else{
      multiChoice = true;
      item.className  += ' over';

      let jObject = new Object();  
      jObject.image_id = item.src.split(/[/]+/).pop();
      jObject.From = "negative";
      jObject.To = "positive";
      jObject.batch = count_num;
      jObject.Time = js_yyyy_mm_dd_hh_mm_ss ();
      jObject.adjective = keyword;

      jQuery.ajaxSettings.traditional = true;
      
      logParam = JSON.stringify(jObject);
      
      $.ajax({
        url : "/getLog",
        type: 'POST',
        data: {"jsonData" : logParam},
        dataType:'json',
        success: function(data) {
          
        },
        error: function(x, e) {
            alert("error");
        }
    });   
    }
  }

  function checkKeyUp(e) {
    //'space bar' key가 떼지면 confirm
      if (e.keyCode == "32"){
        if(confirm_button.disabled == false){
          confirm_click();
    
        }
      }
    }
    
document.addEventListener("keyup", checkKeyUp, false);


//     if(isOver){
//   e.preventDefault();  
//   isMouseDown = true;
//   currentTodo = item;
//   currentList = [];
//   let multi_items = document.getElementsByClassName('over');
//   tempTodo_list = [];
//   for(let i=0;i<multi_items.length;i++){
//     tempTodo_list.push(multi_items[i]);
    
//     let multi_clone = cloneImage(multi_items[i]);
//     if(multi_clone.getAttribute('slot') != item.getAttribute('slot')){
//       $(".img_temp").append(multi_clone);
//       currentList.push(multi_clone)
//     }
//     else{
//       currentList.push(item);
//     }
//   }

//   for(let i=0;i<currentList.length;i++){
//     currentList[i].style.zIndex = "2";  
//     currentList[i].style.filter = "brightness(50%)";
//   }
//   mouseOffset_list = [];
//   for(let i=0;i<currentList.length;i++){
//     mouseOffset_list.push({x: currentList[i].offsetLeft - e.clientX, y: currentList[i].offsetTop - e.clientY});
//     tempTodo_list[i].remove();
//   }
// }
//     else{
//   e.preventDefault();
//   isMouseDown = true;
//   item.style.zIndex = "2";
//   currentList = [];
//   currentList.push(item);
//   currentTodo = item;
//   mouseOffset_list = [];
//   mouseOffset_list.push({x: item.offsetLeft - e.clientX, y: item.offsetTop - e.clientY});
//   tempTodo.remove();
//   item.style.filter = "brightness(50%)";
//     }
  // }




/*---------------------------------------------------------------------*/




class Queue {
  constructor() {
    this._arr = [];
    this.cnt = 0;
  }
  enqueue(item) {
    this._arr.push(item);
    this.cnt += 1;
  }
  dequeue() {
    this.cnt -= 1;
    return this._arr.shift();
  }
  pop(){
    return this._arr[0];
  }
  length(){
    return this.cnt;
  }
}

//백엔드의 변수를 가져오는 함수
function getSyncScriptParams() {
  var scripts = document.getElementsByTagName('script');
  var lastScript = scripts[scripts.length-1];
  var scriptName = lastScript;
  return {
      keyword : scriptName.getAttribute('keyword'),
      images : scriptName.getAttribute('images'),
      user_id : scriptName.getAttribute('user_id'),
      test : scriptName.getAttribute("test"),
      total_num : scriptName.getAttribute("total_num"),
      count_num : scriptName.getAttribute("count_num"),
      positive_attr_list : scriptName.getAttribute("positive_attr_list"),
      negative_attr_list : scriptName.getAttribute("negative_attr_list"),
      cluster : scriptName.getAttribute("cluster"),
      current_cluster : scriptName.getAttribute("current_cluster"),
      label : scriptName.getAttribute("label")
  };
}

//화면 image 로딩 완료하면 confirm 버튼 활성화
$(document).ready(function() {
  var imagesLoaded = 0;
  var totalImages = $('img').length;

  $('img').each(function(idx, img) {
    $('<img>').on('load', imageLoaded).attr('src', $(img).attr('src'));
  });
  function imageLoaded() {
    imagesLoaded++;
    if (imagesLoaded == totalImages) {
      confirm_button.disabled = false;
    }
  }
});

//confirm 누름
function confirm_click(){
  timeEnd = Date.now();
  confirm_button.disabled = true;
  classifyImages();
}

//logout 누름
function logout_click(){
  // window.location = "http://130.211.240.166:5000/logout";
  window.location.href = "http://127.0.0.1:5000/logout";
}  

function classifyImages(){

  // 배치된 image를 jarray형식으로 백엔드(/getData)로 전송
  let todo_list = document.getElementsByClassName("row")[0].getElementsByClassName("todo-item");
  let Jarray = new Array();
  let timeStamp= timeEnd - timeStart;
  console.log(timeStamp);
  timeStamp = JSON.stringify(timeStamp);

  for(let i=0;i<todo_list.length;i++){
    let left_right = 0;
    if(todo_list[i].className.includes("over")){
      left_right = 1;
    }
    else {
      left_right = -1;
    }

    let jObject = new Object();
    jObject.user_id = user_id;
    jObject.cluster = current_cluster[i];
    jObject.image_id = todo_list[i].src.split(/[/]+/).pop();
    jObject.adjective = keyword;
    jObject.label = left_right;
    jObject.time = timeStamp;
    jObject.batch = count_num;
    jObject.Time = js_yyyy_mm_dd_hh_mm_ss ();

    Jarray.push(jObject);
  }

  let outParam = JSON.stringify(Jarray);
  jQuery.ajaxSettings.traditional = true;
  $.ajax({
    url : "/getData",
    type: 'POST',
    data: {"jsonData" : outParam},
    dataType:'json',
    success: function(data) {      
      //NUMBER_OF_ADJECTIVE만큼 실험을 안 했다면 화면 초기화(init)
      //NUMBER_OF_ADJECTIVE만큼 실험을 했다면 로그아웃 (이 때, user db의 isDone 필드가 True로 바뀌며 재접속 불가능)
      if(data['index'] < NUMBER_OF_ADJECTIVE){
        init(data);
      }
      else{
        window.location = "http://127.0.0.1:5000/logIn";
        // window.location = "http://130.211.240.166:5000/logIn";
      }
    },
    error: function(x, e) {
        alert("error");
    }
});
  

};

/* 백엔드 parameter 받는 변수 선언 */
var params = new getSyncScriptParams();

var images = JSON.parse(params.images);
var current_cluster = JSON.parse(params.current_cluster);
var label = JSON.parse(params.label);
var total_num = JSON.parse(params.total_num);
var count_num = JSON.parse(params.count_num);
var positive_attr_list = JSON.parse(params.positive_attr_list);
var negative_attr_list = JSON.parse(params.negative_attr_list);
var keyword =params.keyword;
var user_id = params.user_id;

var timeEnd = 0;
var timeStart = 0;

/* create queues */  
var total_queue = new Queue();
total_queue._arr = Object.values(images);
total_queue.cnt = total_queue._arr.length;
let neutral_queue = [];
let blue_queue = [];
let red_queue = [];


function selectList(total_queue,b,n,r)
  {
    for(var i=0;i<b;i++){
      blue_queue.push(total_queue.dequeue());
    }
    for(var i=0;i<n;i++){
      neutral_queue.push(total_queue.dequeue());
    }
    for(var i=0;i<r;i++){
      red_queue.push(total_queue.dequeue());
    } 
  }

function clearAll(){
  blue_queue = [];
  red_queue = [];
  neutral_queue = [];
  
  $('.todo-item').remove();
}

// function reloadQueue(queue, nextComponents){
//   for(var i=0; i < nextComponents.length; i++){
//     queue.enqueue(nextComponents[i]);
//   }
// }

// function enQueues(blue_queue, blue_list, neutral_queue, neutral_list, red_queue,red_list) {
//   reloadQueue(blue_queue, blue_list);
//   reloadQueue(red_queue, red_list);
//   reloadQueue(neutral_queue, neutral_list);
//   return blue_queue, neutral_queue, red_queue;
// }

function js_yyyy_mm_dd_hh_mm_ss () {
  now = new Date();
  year = "" + now.getFullYear();
  month = "" + (now.getMonth() + 1); if (month.length == 1) { month = "0" + month; }
  day = "" + now.getDate(); if (day.length == 1) { day = "0" + day; }
  hour = "" + now.getHours(); if (hour.length == 1) { hour = "0" + hour; }
  minute = "" + now.getMinutes(); if (minute.length == 1) { minute = "0" + minute; }
  second = "" + now.getSeconds(); if (second.length == 1) { second = "0" + second; }
  return year + "-" + month + "-" + day + " " + hour + ":" + minute + ":" + second;
}

/* array에 있는 image를 보여주는 함수 */
function displayImages(queue){
  onLoadcount = 0;
  for(var i=1;i<=queue.length;i++){

    if(queue[i-1] != null){
      var img_node = document.createElement('img');
      img_node.setAttribute("class","todo-item");
      img_node.src = IMAGE_PATH + queue[i-1];
      var side = "";

      if(queue == blue_queue){
        side = "L";
        img_node.setAttribute("id",0);
      }
      else if(queue == red_queue){
        side = "R";
        img_node.setAttribute("id",2);
      }
      else{
        side = "N";
        img_node.setAttribute("id",1);
      }
      var ID = '#'.concat(side,String(i));
      if($(ID).length != 0){
        $(ID).append(img_node);
      }
      else{
        container = null;
        classname = null;
        if(side == "L"){
          container = document.getElementsByClassName('blue')[0];
          classname = "left";
        }
        else if(side == "R"){
          container = document.getElementsByClassName('red')[0];
          classname = "right";
        }
        else{
          container = document.getElementsByClassName('neutral')[0];
          classname = "center";
        }
        console.log(container);

        let row_count = container.getElementsByClassName('image_row')[0].childElementCount;
        console.log(row_count);
        new_row = document.createElement('div');
          for(let j=0;j<row_count;j++){  
            new_slot = document.createElement('div');
            new_slot.className = classname;
            console.log(ID.replace(side,""));
            new_slot.id = side + (parseInt(ID.replace("#".concat(side),""))+j);
            
            if(j==0){
              new_slot.append(img_node);
              img_node.className = img_node.className.replace(" over","");  
              setListener(img_node);
              img_node.style.left = 0 + "px";
              img_node.style.top = 0 + "px";
              // currentTodo = null;
              // fullCount = 0; 
            }
            new_row.append(new_slot); 
          }
          new_row.className = "image_row";
          container.append(new_row);
        }
      }
      img_node.onload = function(){
        onLoadcount++;
      if(onLoadcount == totalDisplay){
        confirm_button.disabled = false;  
        timeStart = Date.now();
      }};
    }

  }


function init(data){
  clearAll();

  if(typeof data != "undefined"){
    
    blue_queue = data['blue'];
    neutral_queue = data['neutral'];  
    red_queue = data['red'];
    keyword = data['keyword'];
    count_num = data['image_count'];
    isNewset = data['isNewset'];
    score = data['score'];
    current_cluster = data['current_cluster'];
    positive_attr_list = data['positive_attr_list'];
    sorted_positive_attr_list = positive_attr_list.sort(function(x, y){
      return d3.descending(x.score, y.score);
   });
    negative_attr_list = data['negative_attr_list'];
    sorted_negative_attr_list = negative_attr_list.sort(function(x, y){
      return d3.descending(x.score, y.score);
   });
    
    var temp_dots = []
    for(x in dots)
    temp_dots.push(dots[x].image_id);

    for(let i=0;i<score.length;i++){
      score[i].score = parseFloat(dots[temp_dots.indexOf(score[i].image_id)].score) + score[i].score;
      dots[temp_dots.indexOf(score[i].image_id)].score = score[i].score;
    }

    drawBar(sorted_positive_attr_list, sorted_negative_attr_list);
    markLabel(score);
    returnCurrent(beforeLabel);
    currentLabeling(current_cluster);

    if(isNewset){
      returnMark();
    }
  }
  else{
      selectList(total_queue,0,BATCH_NUMBER,0);
  }

  $('.keyword').text(keyword);
  $('.count').text(count_num);
  $('.total').text(total_num);

  totalDisplay = blue_queue.length + neutral_queue.length + red_queue.length;

  
  displayImages(neutral_queue);

  todoItems = document.getElementsByClassName("todo-item");
  setListeners(todoItems);
}

/*--------------------------- tsne 그래프 ----------------------------------------*/

var margin = { top: 0, right: 30, bottom: 0, left: 0},
tsne_width = 350;
tsne_height = 670;
var tsne_svg = d3.select("#tsne_div")
          .append("svg")
          .attr("width", 330 + "px")
          .attr("height",  tsne_height + "px")
          .style("border","none") 
          .style("background-color", "none")
          .call(d3.zoom()
                 .on("zoom", function () {
          tsne_svg.attr("transform", d3.event.transform)
                 })
                 .scaleExtent([1,4])
                 .translateExtent([[0,0],[350,tsne_height]])
            )
          .append("g");

var tsne_svg1 = d3.select('svg')

tsne_svg1.append('rect')
    .attr('x',1)
    .attr('y',tsne_height-100)
    .attr("width",130)
    .attr("height",100)
    .attr("stroke","#151515")
    .attr("stroke-width",1)
    .style("fill","#FFFFFF");


    tsne_img = d3.select('body').append('div')
    .attr("class","tsne_img");

                

// legend 정의
var legend = tsne_svg1.selectAll(".legend")
                 .data([{text:'Positive', color:"#417AFF", border:"transparent"},
                       {text:'Negative', color:'#F26C6C', border:"transparent"},
                      {text: 'Not labeled', color:'#AAAAAA', border:"transparent"},
                      {text: 'Current Image', color:'#FFFFFF', border:"#FF0000"}])
                 .enter().append("g")
                 .attr("class","legend")
                 .attr("transform", function(d, i) {return "translate(0," + i*20 + ")";});

legend.append("circle").attr("cx",15)
                       .attr("cy",tsne_height - 80)
                       .attr("r", 6)
                       .style("fill", function(d){return d.color})
                       .attr("stroke-width", 2)
                       .attr("stroke", function(d){return d.border});


legend.append("text").attr("x", 30)
                     .attr("y", tsne_height - 80)
                     .attr("dy", ".35em")
                     .attr("font-size",13)
                     .text(function(d) { return d.text});

var rect = tsne_svg.append("rect")
.attr("width", tsne_width)
.attr("height", tsne_height)
.style("fill", "none")
.style("pointer-events", "all");

var tsne_container = tsne_svg.append("g");

tsne_container.append("g")
.attr("class", "x axis")
.selectAll("line")
.data(d3.range(0, tsne_width, 10))
.enter().append("line")
.attr("x1", function (d) { return d; })
.attr("y1", 0)
.attr("x2", function (d) { return d; })
.attr("y2", tsne_height);

tsne_container.append("g")
.attr("class", "y axis")
.selectAll("line")
.data(d3.range(0, tsne_height, 10))
.enter().append("line")
.attr("x1", 0)
.attr("y1", function (d) { return d; })
.attr("x2", tsne_width)
.attr("y2", function (d) { return d; });

xList = [];
yList = [];

dots = JSON.parse(params.cluster);

for(let i=0;i<dots.length;i++){
  dots[i].x = parseFloat(dots[i].x);
  xList.push(dots[i].x);
  dots[i].y = parseFloat(dots[i].y);
  yList.push(dots[i].y);
}

function scaleData(data,xList,yList){
  for(let i =0;i<data.length;i++){
    data[i].x = ( data[i].x - d3.min(xList) ) / (d3.max(xList) - d3.min(xList)) * (tsne_width - d3.quantile(xList,0.15));
    data[i].y = ( data[i].y - d3.min(yList) ) / (d3.max(yList) - d3.min(yList)) * (tsne_height - d3.quantile(yList,0.15));
  }
}
scaleData(dots,xList,yList);
var color_scale = d3.scaleLinear()
                    .domain([-1,1])
                    .range(['#F26C6C', "#417AFF"]);


// dots = [{x:127,y:127}, {x:133,y:133} , {x:155,y:155}, {x:156.5,y:156.5}];

var tempStroke = null;

dot = tsne_container.append("g")
    .attr("class", "dot")
    .selectAll("circle")
    .data(dots)
    .enter().append("circle")
    .attr("r", 5)
    .attr("id", function (d) { return d.image_id;})
    .attr("cx", function (d) { return d.x; })
    .attr("cy", function (d) { return d.y; })
    .style("fill","#AAAAAA")
    .style("stroke", "transparent")
    .style("stroke-width", 2)
    .on("mouseover",function(d){
      tempStroke = d3.select(this).style("stroke");
      
      //tsne 클릭 관련
      // d3.select(this).style("cursor","pointer");
      d3.select(this).style("stroke","green");
      
      tsne_img
        .style('opacity',0.9)
        .style('left',d3.event.pageX + "px")
        .style('top',d3.event.pageY + "px");

      tsne_img
      .append('img')
      .transition().duration(500)
      .attr('src',IMAGE_PATH+d.image_id)
      .attr('width',100)
      .attr('height',100);
    })
    .on("mouseout", function(d) {      
      d3.select(this).style("stroke",tempStroke);

      tsne_img.select('img').remove();
      tsne_img.style("opacity", 0);  

    });
    //tsne 클릭 관련
    // .on("click", function(d) {
    //   let jObject = new Object(); 
    //   jObject.image_id = d.image_id;
    //   jObject.type = "tsne";
    //   $.ajax({
    //     url : "/getCurrent",
    //     type: 'POST',
    //     data: jObject,
    //     dataType:'json',
    //     success: function(data) {
    //       init(data);
    //     },
    //     error: function(x, e) {
    //         alert("error");
    //     }
    // });
    // })



function currentLabeling(data){
  for(let i=0;i<data.length;i++){
    var circle = tsne_container.select('[id="'.concat(data[i],'"]'));
   
    circle
        .transition()
        .duration(750)
        .style("stroke", "#FF0000")
        .style("stroke-width", 2);  
  }
  beforeLabel = data;
}

function returnCurrent(data){
  for(let i=0;i<data.length;i++){
    var circle = tsne_container.select('[id="'.concat(data[i],'"]'));
    
    circle
        .transition()
        .duration(750)
        .style("stroke", "transparent")
        .style("stroke-width", 2);
  }
}

function returnMark(){
  var circle = tsne_container.selectAll('circle');
  circle
        .style("fill","#AAAAAA");
}


function drawBar(positive_data, negative_data){
    let axisscale = 0;
  if(typeof positive_data[0] != "undefined" && typeof negative_data[0] != "undefined"){  
    axisscale = Math.max(positive_data[0].score, negative_data[0].score);
  }
  else{
    axisscale = 1;
  }
  xRange1.domain([0,axisscale]);
  xRange2.domain([axisscale,0]);
  bar_svg1.select(".x")
          .transition()
          .call(xAxis1);
  bar_svg2.select(".x")
          .transition()
          .call(xAxis2);

  update(bar_svg1,positive_data, axisscale, "blue");
  update(bar_svg2,negative_data, axisscale, "red");

}

function markLabel(data){
  for(let i=0;i<data.length;i++){
    if(data[i].labeled){
      var circle = tsne_container.select('[id="'.concat(data[i].image_id,'"]'));
      let color = color_scale(parseFloat(data[i]['score']));
      circle
        .style("fill",color);  
    }  
  }
}
/* Attriubute Statistic */

let rect_width = 140;
let rect_x = 20;
let rect_margin = 13;
let axisRange = 579 - rect_margin - rect_width - rect_x;

let scale = 0;

sorted_positive_attr_list = positive_attr_list.sort(function(x, y){
  return d3.descending(x.score, y.score);
});


sorted_negative_attr_list = negative_attr_list.sort(function(x, y){
  return d3.descending(x.score, y.score);
});
if(typeof sorted_positive_attr_list[0] != "undefined" && typeof sorted_negative_attr_list[0] != "undefined"){  

  scale = Math.max(sorted_positive_attr_list[0].score, sorted_negative_attr_list[0].score);

}
else{
  scale = 1;
}

var xRange1 = d3.scaleLinear()
                .range([rect_x, 579 - rect_margin - rect_width])
                .domain([0, scale]);
var xAxis1 = d3.axisTop()
              .scale(xRange1);

var xRange2 = d3.scaleLinear()
              .range([rect_width + rect_x, 579 - rect_margin])
              .domain([scale, 0]);
var xAxis2 = d3.axisTop()
            .scale(xRange2);

// Positive bar attribute

var bar_svg1 = d3.select(".bar_positive").append("svg")
.attr("width", 100 + "%")
.attr("height", 100 + "%")
.append("g");

bar_svg1.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + 30 + ")")
      .call(xAxis1);

var bar_svg2 = d3.select(".bar_negative").append("svg")
.attr("width", 100 + "%")
.attr("height", 100 + "%")
.append("g");

bar_svg2.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + 30 + ")")
      .call(xAxis2);
            
update(bar_svg1,positive_attr_list, scale, "blue");
update(bar_svg2,negative_attr_list, scale, "red");
tempStroke2 = null;
function update(svg,data,scale, blueOrred){
  sorted_data = data.sort(function(x, y){
    return d3.descending(x.score, y.score);
 });

  let nodePos = null;
  let scorePos = null;
  if(blueOrred == "blue"){
    nodePos = 579 - rect_width;
    scorePos = rect_x;
    scoreClass = "blue_score";
  }
  else{
    nodePos = rect_x;
    scorePos = 579 - 60;
    scoreClass = "red_score";
  }

  var text_attribute = svg.selectAll("text.attribute").data(sorted_data);
  text_attribute.enter().append("text").attr("class","attribute")
                .attr("x", nodePos + 5)
                .attr("y", function(d, i) { return 75 + i*40})
                .attr("dy", ".35em")
                .attr("font-size",13)
                .text(function(d) { return d.attribute});

  text_attribute
                        .attr("x", nodePos + 5)
                        .attr("y", function(d, i) { return 75 + i*40})
                        .attr("dy", ".35em")
                        .attr("font-size",13)
                        .text(function(d) { return d.attribute});

  var blue_rect_bar = svg.selectAll('rect.blue_bar').data(sorted_data);
  var red_rect_bar = svg.selectAll('rect.red_bar').data(sorted_data);
if(blueOrred == "blue"){


  var rect_nodes = svg.selectAll('rect.node').data(sorted_data);
  rect_nodes.enter().append("rect").attr("class","node")
                    .attr("id",function(d){ return d.attribute})
                    .attr("x",nodePos)
                    .attr("y",function(d, i){ return 60 + i*40})
                    .attr("width", rect_width-5)
                    .attr("height", 30);
                    // .on("mouseover", function(d){
                    //   d3.select(this).style("cursor","pointer");
                      
                    //   tempStroke = d3.select(this).style("stroke");
                    //   d3.select(this).style("stroke","green");
                    // })
                    // .on("mouseout", function(d) {      
                    //   d3.select(this).style("stroke",tempStroke);
                    // })
                    // .on("click", function(d) {
                    //   let jObject = new Object(); 
                    //   jObject.attribute = d.attribute;
                    //   if( blueOrred == "blue"){
                    //     jObject.label = "positive";
                    //   }
                    //   else{
                    //     jObject.label = "negative";
                    //   }
                    //   jObject.type = "attribute";
                    //   console.log(d.attribute);
                    //   $.ajax({
                    //     url : "/getCurrent",
                    //     type: 'POST',
                    //     data: jObject,
                    //     dataType:'json',
                    //     success: function(data) {
                    //       init(data);
                    //     },
                    //     error: function(x, e) {
                    //         alert("error");
                    //     }
                    // });
                    // });

  rect_nodes
        .attr("id",function(d){ return d.attribute})
        .attr("x",nodePos)
        .attr("y",function(d, i){ return 60 + i*40})
        .attr("width", rect_width-5)
        .attr("height", 30);


  blue_rect_bar.enter().append("rect").attr("class","blue_bar")
                  .attr("x", 20)
                  .attr("y", function(d, i) { return 62 +  i*40})
                  .transition()
                  .attr("width", function(d){ return d.score / scale * axisRange})
                  .attr("height", 26)
                  .style("fill","blue")
                  .style("opacity",0.7);

  blue_rect_bar.transition()
                  .attr("x", 20)
                  .attr("y", function(d, i) { return 62 +  i*40})
                  .attr("width", function(d){ return d.score / scale * axisRange})
                  .attr("height", 26)
                  .style("fill","blue")
                  .style("opacity",0.7);

  blue_rect_bar.exit().remove();
  }
  else{  

    
  var rect_nodes = svg.selectAll('rect.node').data(sorted_data);
  rect_nodes.enter().append("rect").attr("class","node")
                    .attr("id",function(d){ return d.attribute})
                    .attr("x",nodePos)
                    .attr("y",function(d, i){ return 60 + i*40})
                    .attr("width", rect_width-5)
                    .attr("height", 30); 

  rect_nodes
        .attr("id",function(d){ return d.attribute})
        .attr("x",nodePos)
        .attr("y",function(d, i){ return 60 + i*40})
        .attr("width", rect_width-5)
        .attr("height", 30);


  red_rect_bar.enter().append("rect").attr("class","red_bar")
                  .attr("x", function(d){ return xRange2(d.score);})
                  .attr("y", function(d, i) { return 62 +  i*40})
                  .transition()
                  .attr("width", function(d){return xRange2(0) - xRange2(d.score)})
                  .attr("height", 26)
                  .style("fill","red")
                  .style("opacity",0.8);

  red_rect_bar.transition()
                  .attr("x", function(d){ return xRange2(d.score);})
                  .attr("y", function(d, i) { return 62 +  i*40})
                  .attr("width", function(d){return xRange2(0) - xRange2(d.score)})
                  .attr("height", 26)
                  .style("fill","red")
                  .style("opacity",0.8);
                  
  red_rect_bar.exit().remove();
  }


  var bar_score = svg.selectAll('text.'.concat(scoreClass)).data(sorted_data);

  bar_score.enter().append("text").attr("class",scoreClass)
  .attr("x", scorePos)
  .attr("y", function(d, i ) { return 75 + i*40})
  .attr("dy", ".35em")
  .attr("font-size",13)
  .style("fill","#FFFFFF")
  .text(function(d) {return (d.score * 100).toFixed(1) + "%"});

  bar_score
      .attr("x", scorePos)
      .attr("y", function(d, i ) { return 75 + i*40})
      .attr("dy", ".35em")
      .attr("font-size",13)
      .style("fill","#FFFFFF")
      .text(function(d) {return (d.score * 100).toFixed(1) + "%"});

  bar_score.exit().remove();

  rect_nodes.exit().remove();
  text_attribute.exit().remove();

      }      


/* main */
init();
currentLabeling(current_cluster);
markLabel(dots);


