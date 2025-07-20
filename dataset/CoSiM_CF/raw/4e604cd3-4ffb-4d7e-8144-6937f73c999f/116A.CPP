#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int n, exit, enter, tmp = 0;
  int capacity = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> exit >> enter;
    tmp += enter - exit;
    if (tmp > capacity)
      capacity = tmp;
  }
  cout << capacity;
  return 0;
}
