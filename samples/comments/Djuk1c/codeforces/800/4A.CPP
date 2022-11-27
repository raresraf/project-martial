#include <iostream>
#include <string>
using namespace std;
int main() {
  int num;
  string ret = "";
  cin >> num;
  if (num == 2) {
    ret = "NO";
    cout << ret;
    return 0;
  }
  if (num % 2 == 0) {
    ret = "YES";
  } else {
    ret = "NO";
  }
  cout << ret;
  return 0;
}
