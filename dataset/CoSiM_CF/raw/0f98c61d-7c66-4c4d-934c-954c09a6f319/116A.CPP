#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
  int n;
  int answer = 0;
  int maxim = 0;
  int arriving = 0;
  cin >> n;
  while (n--) {
    int entering, exiting;
    cin >> exiting >> entering;
    arriving = arriving - exiting + entering;
    if (arriving > maxim) {
      maxim = arriving;
    }
  }
  cout << maxim;
}