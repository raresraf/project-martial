#include <iostream>

using namespace std;

const unsigned short n_friend = 3, threhold = 2;

int main() {
  // get number of problems
  register unsigned short n = 0, i = 0, j = 0, temp = 0, notSure = 0,
                          noSolution = 0;
  cin >> n;
  // read the matrix
  for (i = 0; i < n; i++) {
    notSure = 0;
    for (j = 0; j < n_friend; j++) {
      cin >> temp;
      if (temp == 0) {
        notSure++;
      }
    }
    if (threhold + notSure > n_friend) {
      noSolution++;
    }
  }
  // output yes solution
  cout << n - noSolution << endl;
  return 0;
}