//
//  main.cpp
//  339A
//
//  Created by Alejandro Gleason on 4/6/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  char foo[105];
  int count = 0;
  int strx[105];
  // scanf("%s",strx);
  // printf("%s",&strx);
  getline(cin, str);
  // cout << str.length();
  // cout << str[0] << endl;
  // int n = str.length();
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '1' || str[i] == '2' || str[i] == '3') {
      // cout << str[i];
      foo[count] = str[i];
      count++;
    }
  }
  sort(foo, foo + count);

  if (str.length() == 1) {
    printf("%c", foo[0]);
    return 0;
  }

  for (int i = 0; i < count; i++) {
    printf("%c", foo[i]);
    if (i != count - 1) {
      printf("%c", '+');
    }
  }
  return 0;
}
