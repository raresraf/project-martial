// my first program in C++
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  for (size_t i = 0; i < n; i++) {
    string temp;
    cin >> temp;
    size_t len = temp.length();
    if (len <= 10) {
      cout << temp << endl;
    } else {
      cout << temp[0] << len - 2 << temp[len - 1] << endl;
    }
  }
  return 0;
}