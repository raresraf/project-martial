#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int n;
  char now, pre;
  int take_num = 0;
  cin >> n;
  cin >> pre;
  for (int i = 1; i < n; ++i) {
    cin >> now;
    if (now == pre)
      ++take_num;
    else
      pre = now;
  }
  cout << take_num;
  return 0;
}
