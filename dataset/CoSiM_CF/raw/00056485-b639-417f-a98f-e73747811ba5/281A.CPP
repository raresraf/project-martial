#include <iostream>

using namespace std;

int main() {

  string str;
  cin >> str;

  int up_upper = (int)('Z');
  int up_lower = (int)('A');
  int low_upper = (int)('z');
  int low_lower = (int)('a');
  char first = str[0];

  if ((int)(first) <= low_upper && (int)(first) >= low_lower) {

    str[0] = (char)(int(first) - (low_lower - up_lower));
  }

  cout << str;
  return 0;
}