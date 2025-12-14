#include <algorithm>
#include <iostream>
using namespace std;
void solved() {
  int a, b;
  int n;
  cin >> n;
  int temp = 0;
  int MAX = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    temp = temp + b - a;
    MAX = max(temp, MAX);
  }
  cout << MAX;
}
int main() {
  solved();
  return 0;
}
