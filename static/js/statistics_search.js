function changeValue() {
  var condition = document.getElementById("search_condition");
  $(".condition_box").removeClass("active");

  if (condition.options[condition.selectedIndex].value == "team") {
    $(".team_search_box").addClass("active");
  } else if (condition.options[condition.selectedIndex].value == "distribute") {
    $(".distribute_search_box").addClass("active");
  }
}
