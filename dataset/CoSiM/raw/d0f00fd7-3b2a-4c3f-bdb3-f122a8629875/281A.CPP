#include "bits/stdc++.h"
using namespace std;

int main(int argc, char const *argv[]) {
  string str;
  cin >> str;

  str[0] = (tolower(str[0]) - 'a' + 'A');
  cout << str;
  return 0;
}
