/*마우스 기능 */
let itemsContainer = document.getElementById("items-container");
let todosContainer = document.getElementById("todos-container");

let mouseOffset = {x:0, y:0};
let isMouseDown = false;
let currentTodo = null;

let doElsCollide = function(el1, el2) { 
     el1.offsetBottom= el1.offsetTop + el1.offsetHeight;
     el1.offsetRight = el1.offsetLeft + el1.offsetWidth;
     el2.offsetBottom = el2.offsetTop + el2.offsetHeight;
     el2.offsetRight = el2.offsetLeft + el2.offsetWidth;

     return !((el1.offsetBottom < el2.offsetTop) ||
               (el1.offsetTop > el2.offsetBottom) ||
               (el1.offsetRight < el2.offsetLeft) ||
               (el1.offsetLeft > el2.offsetRight))
};


function onMouseDown(e, item) {
  isMouseDown = true;
  currentTodo = item;

  mouseOffset = {x: item.offsetLeft - e.clientX, y: item.offsetTop - e.clientY};
  
  item.style.filter = "brightness(50%)";
}

function onMouseMove(e) {
  e.preventDefault();  
  if(isMouseDown) {
    currentTodo.style.left = e.clientX + mouseOffset.x + "px";
    currentTodo.style.top = e.clientY + mouseOffset.y + "px";
    }
}

function onMouseUp(e, item) {
  isMouseDown = false;
  item.style.filter = "brightness(100%)";
}

function onMouseOver(e, item) {
  if(!isMouseDown){
    item.style.filter = "brightness(130%)";
  }
}
function onMouseOut(e, item) {
  if(!isMouseDown){
    item.style.filter = "brightness(100%)";
  }
}

function setListeners(todoItems) {
  for(let i = 0; i < todoItems.length; i++) {
  let item = todoItems[i];
  item.addEventListener("mousedown", (e) => { onMouseDown(e, item); });
  item.addEventListener("mouseover", (e) => { onMouseOver(e, item); });
  item.addEventListener("mouseout", (e) => { onMouseOut(e, item); });
  document.body.addEventListener("mousemove", (e) => {
    onMouseMove(e);
  });
  item.addEventListener("mouseup", (e) => {
    onMouseUp(e, item);
  });
  
}}

function setListener(todoItem) {
  
  todoItem.addEventListener("mousedown", (e) => { onMouseDown(e, todoItem); });
  todoItem.addEventListener("mouseover", (e) => { onMouseOver(e, todoItem); });
  todoItem.addEventListener("mouseout", (e) => { onMouseOut(e, todoItem); });
  document.body.addEventListener("mousemove", (e) => {
    onMouseMove(e);
  });
  todoItem.addEventListener("mouseup", (e) => {
    onMouseUp(e, todoItem);
  });
  
}

setInterval(() => {
  let areas = document.getElementsByClassName("red-blue");
  for(let i = 0; i < areas.length; i++) {
       
    areas[i].className = areas[i].className.replace("over", "");
    if(doElsCollide(currentTodo, areas[i])) {
      areas[i].className += " over"; 
      if(!isMouseDown) {
        snapTodo(currentTodo, areas[i], i);
      }
    }
  }
}, 100);

function snapTodo(todo, container,index) {
  area_list = ["left","right"];
    temp_list = document.getElementsByClassName(area_list[index]);
    for(let i=0;i<temp_list.length;i++){
      let item = temp_list[i];
      if(!item.hasChildNodes()){
        let box = item.getBoundingClientRect();
        todo_clone = todo.cloneNode();
        todo.remove();
        item.append(todo_clone);
        setListener(todo_clone);
        todo_clone.style.left = box.x + "px";
        todo_clone.style.top = box.y - 10 + "px";
        currentTodo = null;
        

        break;
      }
    }

  }




/*------------------------------------------------------------------*/

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

function confirm_click(){
  // 경민이형 여기에 ajax 기능 추가해줘
  if(total_queue.cnt > 0){
  init();
  }
  else{
    alert("done!")
  }
}

function getSyncScriptParams() {
   var scripts = document.getElementsByTagName('script');
   var lastScript = scripts[scripts.length-1];
   var scriptName = lastScript;
   return {
       keywords : scriptName.getAttribute('keywords'),
       images : scriptName.getAttribute('images'),
   };
 }

var blue_test_number = 6;
var red_test_number = 6;
var neutral_test_number = 2;
var params = new getSyncScriptParams();
var images = JSON.parse(params.images);
  
/* create queues */  
var total_queue = new Queue();
total_queue._arr = Object.values(images);
total_queue.cnt = total_queue._arr.length;
const neutral_queue = new Queue();
const blue_queue = new Queue();
const red_queue = new Queue();

function description(){
  var keyword_list = JSON.parse(params.keywords);
  var left_word = "more likely "+ keyword_list[0];
  var right_word = "more likely not " + keyword_list[0];
  $('#adjective1').append(left_word);
  $('#adjective2').append(right_word);
  var sec_description = "Images in the left box will be labeled as \"" + keyword_list[0] +
  "\", in the middle box as \"NEUTRAL\", and in the right box as \"not "+ keyword_list[0]+"\"."
  $('#description2').append(sec_description);
}

function selectList(total_queue,b,n,r)
  {
    blue_temp = []
    neutral_temp = []
    red_temp = []
    
    for(var i=0;i<b;i++){
      blue_temp.push(total_queue.dequeue());
    }
    for(var i=0;i<n;i++){
      neutral_temp.push(total_queue.dequeue());
    }
    for(var i=0;i<r;i++){
      red_temp.push(total_queue.dequeue());
    }
    
    return [blue_temp, neutral_temp, red_temp]
  }
/* selectList에서 연산 결과값 반환*/

  
function clearQueue(queue){
  queue._arr = [];
  queue.cnt = 0;
}


function clearAll(){
  clearQueue(blue_queue);
  clearQueue(neutral_queue);
  clearQueue(red_queue);
  
  $('.todo-item').remove();
}

function reloadQueue(queue, nextComponents){
  for(var i=0; i < nextComponents.length; i++){
    queue.enqueue(nextComponents[i]);
  }
}

function enQueues(blue_queue, blue_list, neutral_queue, neutral_list, red_queue,red_list) {
  /* 나중에 백엔드 구축 나눠야함*/
  reloadQueue(blue_queue, blue_list);
  reloadQueue(red_queue, red_list);
  reloadQueue(neutral_queue, neutral_list);
  return blue_queue, neutral_queue, red_queue;
}


function displayImages(queue){
    
  for(var i=1;i<=queue.cnt;i++){
    if(queue._arr[i-1] != null){
      var img_node = document.createElement('img');
      img_node.setAttribute("class","todo-item");
      img_node.src = 'static/image/FFHQ_SAMPLE/' + queue._arr[i-1];
      var side = ""

      if(queue == blue_queue){
        side = "L"
      }
      else if(queue == red_queue){
        side = "R"
      }
      else{
        side = "N"
      }
      var ID = '#'.concat(side,String(i))
      $(ID).append(img_node);
    }
  }
}


function init(){
  clearAll();
  var getLists = new selectList(total_queue,blue_test_number,neutral_test_number,red_test_number);
  var blue_list = getLists[0];
  var neutral_list = getLists[1];
  var red_list = getLists[2];
  enQueues(blue_queue,blue_list,neutral_queue,neutral_list, red_queue,red_list);

  displayImages(blue_queue);
  displayImages(red_queue);
  displayImages(neutral_queue);
  
  todoItems = document.getElementsByClassName("todo-item");
  setListeners(todoItems);

}


description();
init();



   setInterval(() => {
     let snaps = document.getElementsByClassName("snap");
     for(let i = 0; i < snaps.length; i++) {
          
       //Clear the Over class every time (Hide Elements are not under collision)
       snaps[i].className = snaps[i].className.replace("over", "");
       if(doElsCollide(currentTodo, snaps[i])) {
         //There is a collision then we are good to
         snaps[i].className += " over"; ///< Over class will show the snap container 
     
         if(!isMouseDown) {
          //Snap Current Todo under Current Snap Container :)
          snapTodo(currentTodo, snaps[i]);
          
        }
     }
     }
  }, 100);



