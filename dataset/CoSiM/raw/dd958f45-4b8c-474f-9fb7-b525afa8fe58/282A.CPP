#include <iostream>
#include <vector>
using namespace std;
int main() {
  int x = 0;
  string a;
  vector<string> v;
  cin >> x;
  while (x--) {
    cin >> a;
    v.push_back(a);
  }
  x = 0;
  for (int i = 0; i < v.size(); i++) {
    if (v[i] == "x++" || v[i] == "X++" || v[i] == "++x" || v[i] == "++X")
      x++;
    else
      x--;
  }
  cout << x;
  return 0;
}
