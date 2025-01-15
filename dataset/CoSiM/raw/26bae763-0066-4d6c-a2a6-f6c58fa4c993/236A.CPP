#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string str = " ";
  cin >> str;
  sort(str.begin(), str.end());
  str.erase(unique(str.begin(), str.end()), str.end());
  if (str.length() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else if (str.length() % 2 != 0) {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
