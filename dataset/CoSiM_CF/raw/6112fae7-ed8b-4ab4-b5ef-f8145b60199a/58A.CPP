#include <iostream>
using namespace std;
int main() {

  string result = "hello";
  int pass = 0;
  string txt;
  cin >> txt;

  int i = 0;
  while (i < txt.length() && pass < result.length()) {
    txt[i] == result[pass] ? pass++ : i;
    i++;
  }

  pass == result.length() ? cout << "YES" : cout << "NO";

  return 0;
}
