/*
 * stringtask.cpp
 * http://codeforces.com/problemset/problem/118/A
 *  Created on: Oct 22, 2017
 *      Author: athena
 * 31609597 	GNU C++14 	Accepted 	30 ms 	1880 KB
 * 2017-10-22 19:51:17 	2017-10-22 19:51:17
 */

#include <iostream>
#include <locale>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;

  locale loc;

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == 'A' || str[i] == 'O' || str[i] == 'Y' || str[i] == 'E' ||
        str[i] == 'U' || str[i] == 'I' || str[i] == 'a' || str[i] == 'o' ||
        str[i] == 'y' || str[i] == 'e' || str[i] == 'u' || str[i] == 'i') {
      cout << "";
    } else if (isupper(str[i])) {
      cout << "." << tolower(str[i], loc);
    } else {
      cout << "." << str[i];
    }
  }

  return 0;
}
