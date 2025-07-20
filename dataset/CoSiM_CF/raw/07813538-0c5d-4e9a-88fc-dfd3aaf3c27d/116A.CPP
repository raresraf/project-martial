#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int max, curr;
  max = 0;
  curr = 0;
  int n;
  cin >> n;
  while (n--) {
    int a, b;
    cin >> a >> b;
    curr = curr - a;
    curr = curr + b;
    if (curr > max) {
      max = curr;
    }
  }
  cout << max;
  return 0;
}