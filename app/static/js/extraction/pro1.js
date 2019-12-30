var address = 'http://localhost:9999';
// var address = 'http://49.234.19.31:8082';
// var address = 'http://111.229.161.111:8082';


$(function(){
	// alert("1111");
	// 
	var sSaying = "";
	var mydict={"name":"Graph","children":[]};
	// var mydict = {"name":"中国",
	// 			  "children":
	// 			  [
	// 			    {
	// 			      "name":"浙江" ,
	// 			      "children":
	// 			      [
	// 			        {"name":"杭州" },
	// 			        {"name":"宁波" },
	// 			        {"name":"温州" },
	// 			        {"name":"绍兴" }
	// 			      ]
	// 			    },
				 
	// 			    {
	// 			      "name":"广西" ,
	// 			      "children":
	// 			      [
	// 			        {
	// 			          "name":"桂林",
	// 			          "children":
	// 			          [
	// 			            {"name":"秀峰区"},
	// 			            {"name":"叠彩区"},
	// 			            {"name":"象山区"},
	// 			            {"name":"七星区"}
	// 			          ]
	// 			        },
	// 			        {"name":"南宁"},
	// 			        {"name":"柳州"},
	// 			        {"name":"防城港"}
	// 			      ]
	// 			    }

	// 			  ]
	// 			}
	var oPop = $('#pop_con');
	var iTime = 100;
	oPop.css({display:'block'});
	var timer = setInterval(function(){					
		iTime--;
		$('.progress-bar').css('width',100-iTime+'%');
		if(iTime==0)
		{
			oPop.css({display:'none'});
			clearInterval(timer);
			iTime=5;
		}
	},180);


	$("input").eq(0).click(function(){
		$.ajax({
			url: address + '/GetContent/mysql',
			type:'GET',
			cache: false,
			dataType:'json',
			success:function(result){
				$("textarea").val(result['content']);
			}
		})
	})


	$("#submit").click(function(){
		var data = $("textarea").val();
		var req_json = JSON.stringify(data);
		var temp_dict = {};
		mydict={"name":"Graph","children":[]};
		$.ajax({
			url: address + '/Extraction/solve',
			type:'post',
			data: req_json,
			contentType: "application/json",
			dataType:'json',
		})
		.done(function(dat){
			sSaying = "";
			console.log(dat)
			for(name in dat){
				if(typeof(temp_dict[dat[name][0]]) == "undefined"){
				temp_dict[dat[name][0]]={"name":dat[name][0],"children":[]}};
				temp_dict[dat[name][0]].children.push({"name":dat[name][2]});

				sSaying += "<tr><th scope=\"row\">"+dat[name][0]+"</th><td>";
				sSaying += dat[name][1] + "</td><td>";
				sSaying += dat[name][2]
				sSaying += "</td></tr>";
			}
			$(".nav-stacked > li").eq(0).click();

			for(key in temp_dict){mydict.children.push(temp_dict[key])};
		})
		
	})

	$("#reset").click(function(){
		$("textarea").val('');

	})


	$(".nav-stacked > li").eq(0).click(function(){
		var cur = $(".nav-stacked > li .active").text();
		if (cur != "言论提取表"){
			$(".showpalce").html("");
			$(".nav-stacked > li").removeClass('active');
			$(this).addClass('active');	

			var temp = "";
			temp = "<table class=\"table\"><thead><tr><th style=\"width:12%\"> \
				人物</th><th style=\"width:8%\">动作</th><th>言论</th></tr></thead><tbody>"
			temp = temp + sSaying +"</tbody></table>"
			$(".showpalce").html(temp);}
		return false;
	})

	$(".nav-stacked > li").eq(1).click(function(){
		var cur = $(".nav-stacked > li .active").text();
		if (cur != "言论提取树形图"){
			$(".showpalce").html("");
			$(".nav-stacked > li").removeClass('active');
			$(this).addClass('active');}

			if(d3.select('svg').length > 0){d3.select('svg').remove()};
			Graphshow(mydict);

		return false;
	})

	$(".nav-stacked > li").eq(2).click(function(){
		var cur = $(".nav-stacked > li .active").text();
		if (cur != "关键词提取"){
			$(".showpalce").html("");
			$(".nav-stacked > li").removeClass('active');
			$(this).addClass('active');}
		return false;
	})

	function Graphshow(treeData){
		console.log(treeData)
		var margin = {top: 10, right: 30, bottom: 10, left:50},
	    width = 960 - margin.right - margin.left,
	    height = 450 - margin.top - margin.bottom;
		    
		var i = 0,duration = 750,root;
		//定义数据转换函数
		var tree = d3.layout.tree()
		    .size([height, width]);
		//定义对角线生成器diagonal
		var diagonal = d3.svg.diagonal()
		    .projection(function(d) { return [d.y, d.x]; });
		//定义svg
		var svg = d3.select("#mytree").append("svg")
			.attr("width", width + margin.right + margin.left)
			.attr("height", height + margin.top + margin.bottom)
			.append("g")
			.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

		root = treeData;
		root.x0 = height / 2;
		root.y0 = 0;

		update(root);
		d3.select(self.frameElement).style("height", "450px");


		function update(source) {
		  // Compute the new tree layout.
		  var nodes = tree.nodes(root).reverse(),
		      links = tree.links(nodes);

		  // Normalize for fixed-depth.控制横向深度
		  nodes.forEach(function(d) { d.y = d.depth * 100; });

		  // Update the nodes鈥 
		  var node = svg.selectAll("g.node")
		      .data(nodes, function(d) { return d.id || (d.id = ++i);});

		  // Enter any new nodes at the parent's previous position.
		  var nodeEnter = node.enter().append("g")
		      .attr("class", "node")
		      .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
		      .on("click", click);

		  nodeEnter.append("circle")
		      .attr("r", 1e-6)
		      .style("fill", function(d) { return d._children ? "#ccff99" : "#fff"; });

		  nodeEnter.append("text")
		      .attr("x", function(d) { return d.children || d._children ? -13 : 13; })
		      .attr("dy", ".35em")
		      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
		      .text(function(d) { return d.name; })
		      .style("fill-opacity", 1e-6)
		     .attr("class", function(d) {
		              if (d.url != null) { return 'hyper'; } 
		         })
		          .on("click", function (d) { 
		              $('.hyper').attr('style', 'font-weight:normal');
		              d3.select(this).attr('style', 'font-weight:bold');
		              if (d.url != null) {
		                 //  window.location=d.url; 
		                 $('#vid').remove();

		                 $('#vid-container').append( $('<embed>')
		                    .attr('id', 'vid')
		                    .attr('src', d.url + "?version=3&amp;hl=en_US&amp;rel=0&amp;autohide=1&amp;autoplay=1")
		                    .attr('wmode',"transparent")
		                    .attr('type',"application/x-shockwave-flash")
		                    .attr('width',"100%")
		                    .attr('height',"100%") 
		                    .attr('allowfullscreen',"true")
		                    .attr('title',d.name)
		                  )
		                }
		          });

		  // Transition nodes to their new position.
		  var nodeUpdate = node.transition()
		      .duration(duration)
		      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

		  nodeUpdate.select("circle")
		      .attr("r", 10)
		      .style("fill", function(d) { return d._children ? "#ccff99" : "#fff"; });

		  nodeUpdate.select("text")
		      .style("fill-opacity", 1);

		  // Transition exiting nodes to the parent's new position.
		  var nodeExit = node.exit().transition()
		      .duration(duration)
		      .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
		      .remove();

		  nodeExit.select("circle")
		      .attr("r", 1e-6);

		  nodeExit.select("text")
		      .style("fill-opacity", 1e-6);

		  // Update the links鈥 
		  var link = svg.selectAll("path.link")
		      .data(links, function(d) { return d.target.id; });

		  // Enter any new links at the parent's previous position.
		  link.enter().insert("path", "g")
		      .attr("class", "link")
		      .attr("d", function(d) {
		        var o = {x: source.x0, y: source.y0};
		        return diagonal({source: o, target: o});
		      });

		  // Transition links to their new position.
		  link.transition()
		      .duration(duration)
		      .attr("d", diagonal);

		  // Transition exiting nodes to the parent's new position.
		  link.exit().transition()
		      .duration(duration)
		      .attr("d", function(d) {
		        var o = {x: source.x, y: source.y};
		        return diagonal({source: o, target: o});
		      })
		      .remove();

		  // Stash the old positions for transition.
		  nodes.forEach(function(d) {
		    d.x0 = d.x;
		    d.y0 = d.y;
		  });
		}

		// Toggle children on click.
		function click(d) {
		  if (d.children) {
		    d._children = d.children;
		    d.children = null;
		  } else {
		    d.children = d._children;
		    d._children = null;
		  }
		  update(d);
		}
	}
})
