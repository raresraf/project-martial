#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int answer = 0;

  for (int i = 0; i < n; i++) {
    int a1, a2, a3;
    cin >> a1 >> a2 >> a3;

    int num_0 = 0;
    int num_1 = 0;

    if (a1 == 1) {
      num_1++;
    }
    if (a2 == 1) {
      num_1++;
    }
    if (a3 == 1) {
      num_1++;
    }

    if (num_1 >= 2) {
      answer++;
    }
  }
  cout << answer;
}