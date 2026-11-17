
---
- [x] primitive values store data directly; reference variables point to objects
	```java
	int age = 25;             // primitive
	boolean active = true;    // primitive
	String name = "Hasan";   // reference
	int[] scores = {90, 80};  // reference
	// `byte`, `short`, `int`, `long`, `float`, `double`, `char`, `boolean`
	```

- [x] IF Conditional
	```java
	int x = 10;
	
	if (x > 10) {
	  System.out.println("x is greater than 10");} 
	else if (x == 10) {
	  System.out.println("x is exactly 10");} 
	else {
	  System.out.println("x is less than 10");}
	```

- [x] SWITCH
	```java
	int day = 1;
	switch (day) {
	  case 1:
	    System.out.println("Monday");
	    break;
	  case 2:
	    System.out.println("Tuesday");
	    break;
	  default:
	    System.out.println("Other day");
	}
	```

- [x] WHILE
	```java
	// normal while loop
	int i = 1;
	while (i <= 3) {
	  System.out.println("Count: " + i);
	  i++;
	}
	
	// do-while syntax
	int j = 1;
	do {
	  System.out.println("Count: " + j);
	  j++;
	} while (j <= 3);
	```

- [x] FOR 
	```java
	// normal for loop
	for (int k = 1; k <= 3; k++) {
	  System.out.println("Count: " + k);}
	
	// enhanced for loop
	String[] cars = {"Volvo", "BMW", "Ford"};
	for (String car : cars) {
	  System.out.println(car);}
	```

---
- [ ] two-dimensional arrays use row and column indexes
	```java
	int[][] grid = {
	  {1, 2},
	  {3, 4}
	};
	
	int value = grid[1][0]; // 3
	```

- [ ] StringBuilder
	```java
	// `String` is immutable; repeated changes create new objects, so use `StringBuilder` when building text in a loop
	StringBuilder result = new StringBuilder();
	
	for (String word : words) {
	  result.append(word).append(" ");
	}
	
	String sentence = result.toString().trim();
	```
