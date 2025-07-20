#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int stops, exit, enter, current, capacity;

  cin >> stops >> exit >> enter;
  stops--;
  current = enter;
  capacity = enter;

  while (stops--) {
    cin >> exit >> enter;

    current -= exit;
    current += enter;

    if (current > capacity) {
      capacity = current;
    }
  }

  cout << capacity;
  return 0;
}
