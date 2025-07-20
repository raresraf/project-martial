#include <bits/stdc++.h>
using namespace std;

int TeamCpp(int number) {
  int res = 0;
  for (int i = 0; i < number; i++) {
    int array[3];
    int count = 0;
    for (int j = 0; j < 3; j++) {
      cin >> array[j];
      if (array[j] == 1) {
        count++;
      }
    }
    if (count >= 2) {
      res++;
    }
  }

  return res;
}

int main() {
  int n;
  cin >> n;
  int result = TeamCpp(n);
  cout << result << endl;

  return 0;
}
