// Codeforces
// 112A - Petya and strings

#include <iostream>
using namespace std;

int main() {
  string sOne;
  cin >> sOne;
  string sTwo;
  cin >> sTwo;

  int difference = 0;

  for (unsigned int i = 0; i < sOne.size(); i++) {
    if (tolower(sOne[i]) > tolower(sTwo[i])) {
      difference = 1;
      break;
    }

    if (tolower(sOne[i]) < tolower(sTwo[i])) {
      difference = -1;
      break;
    }
  }

  cout << difference;

  return 0;
}
