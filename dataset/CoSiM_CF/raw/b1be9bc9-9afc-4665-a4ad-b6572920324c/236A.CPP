#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string inp;
  int count, flag;
  while (cin >> inp) {
    count = 0;
    flag = 0;
    for (int i = 0; i < inp.size(); i++) {
      flag = 0;
      for (int j = 0; j < inp.size(); j++) {

        if (i == j) {
          continue;
        }

        if (inp[i] == inp[j] && i < j) {
          flag = 1;
        }
      }
      if (flag == 0) {
        count++;
      }
    }
    if (count % 2 != 0) {
      cout << "IGNORE HIM!" << endl;
    } else {
      cout << "CHAT WITH HER!" << endl;
    }
  }
  return 0;
}
