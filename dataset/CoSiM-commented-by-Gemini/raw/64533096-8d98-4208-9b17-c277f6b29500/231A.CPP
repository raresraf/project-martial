#include <iostream>
#include <vector>
using namespace std;
int n, e;
int solved;
int result = 0;
vector<int> v;
int main() {
  cin >> n;
  for (int i = 0; i < n; i++) {
    solved = 0;
    for (int i = 0; i < 3; i++) {
      cin >> e;
      if (e == 1) {
        solved++;
      }
    }
    if (solved >= 2) {
      result++;
    }
  }
  cout << result;
  return 0;
}
