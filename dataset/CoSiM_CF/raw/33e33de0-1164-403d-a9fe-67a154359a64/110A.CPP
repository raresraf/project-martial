//
//  110A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>

using namespace std;

int main() {
  long n, count = 0;

  cin >> n;
  string str = to_string(n), ans = "NO";

  size_t found = str.find('4') || str.find('7');

  if (!(found == string::npos)) {
    for (const char &c : str) {
      if (c == '4' || c == '7') {
        count++;
      }
    }
  }

  if (count == 4 || count == 7) {
    ans = "YES";
  }

  cout << ans;

  return 0;
}
