// CodeForces
// Problem 112A [Petya and Strings]

#include <iostream>
using namespace std;

int main() {
  string str1, str2;
  cin >> str1 >> str2;

  for (unsigned int i = 0; i < str1.length(); i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);
  }

  if (str1.compare(str2) > 0) {
    cout << "1";
  } else if (str1.compare(str2) < 0) {
    cout << "-1";
  } else {
    cout << "0";
  }
  return 0;
}