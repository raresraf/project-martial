#include <iostream>
#include <string>
using namespace std;
int main() {
  int x = 0;
  int n;
  string task;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> task;
    if (task == "X++" || task == "++X")
      x++;
    else
      x--;
  }
  cout << x << endl;
  return 0;
}