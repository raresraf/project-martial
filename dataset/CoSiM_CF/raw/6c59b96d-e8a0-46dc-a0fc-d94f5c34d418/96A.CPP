#include "algorithm"
#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string str;
  cin >> str;
  bool succ = false;

  for (int i = 0; i < str.length(); i++) {
    int zeros = count(str.begin() + i, str.begin() + i + 7, '0');
    int ones = count(str.begin() + i, str.begin() + i + 7, '1');
    if (zeros >= 7 || ones >= 7)
      succ = true;
  }

  if (!succ)
    cout << "NO";
  else
    cout << "YES";

  return 0;
}
