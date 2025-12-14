#include <iostream>
#include <string>
using namespace std;
int main() {
  string x;
  int i = 0, arr[5];
  cin >> x;

  i = x.find_first_of('h', i);
  arr[0] = i;
  i = x.find_first_of('e', i);
  arr[1] = i;
  i = x.find_first_of('l', i);
  arr[2] = i;
  i = x.find_first_of('l', i);
  if (i != arr[2]) {
    arr[3] = i;
  } else {
    i = x.find_first_of('l', i + 1);
    arr[3] = i;
  }
  i = x.find_first_of('o', i);
  arr[4] = i;
  for (i = 0; i < 5; i++) {
    if (arr[i] == -1) {
      cout << "NO";
      return 0;
    }
  }
  cout << "YES";
  return 0;
}
