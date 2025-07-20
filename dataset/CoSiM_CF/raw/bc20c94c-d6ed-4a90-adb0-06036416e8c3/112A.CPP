#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str1, str2;
  int result;
  cin >> str1;
  cin >> str2;

  transform(str1.begin(), str1.end(), str1.begin(), ::toupper);
  transform(str2.begin(), str2.end(), str2.begin(), ::toupper);

  if (str1 < str2) {

    result = -1;
  } else if (str1 > str2) {
    result = 1;

  } else {
    result = 0;
  }
  cout << result << endl;
}
