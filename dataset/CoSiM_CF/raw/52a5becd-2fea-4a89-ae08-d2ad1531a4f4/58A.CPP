//
//  main.cpp
//  58A
//
//  Created by Alejandro Gleason on 4/6/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int i = 0, count = 0;
  char stringz[110];
  scanf("%s", stringz);
  int size = strlen(stringz);
  for (; i < size; i++) {
    if (stringz[i] == 'h') { // checking order
      count++;
      i++;
      break; // exiting the loop
    }
  }
  for (; i < size; i++) {
    if (stringz[i] == 'e') { // checking order
      count++;
      i++;
      break; // exiting the loop
    }
  }
  for (; i < size; i++) {
    if (stringz[i] == 'l') { // checking order
      count++;
      i++;
      break; // exiting the loop
    }
  }
  for (; i < size; i++) {
    if (stringz[i] == 'l') { // checking order
      count++;
      i++;
      break; // exiting the loop
    }
  }
  for (; i < size; i++) {
    if (stringz[i] == 'o') { // checking order
      count++;
      i++;
      break; // exiting the loop
    }
  }

  if (count >= 5) { // condition checks that min lenght is accomplished
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
