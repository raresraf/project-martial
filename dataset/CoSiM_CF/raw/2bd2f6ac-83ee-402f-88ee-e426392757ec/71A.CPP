#include <cstring>
#include <iostream>
using namespace std;

int main() {

  int testcase;
  string word;
  cin >> testcase;

  while (testcase--) {

    cin >> word;
    int l = word.length();

    if (l > 10) {
      cout << word.at(0) << l - 2 << word.at(l - 1) << endl;
    } else
      cout << word << endl;
  }

  return 0;
}