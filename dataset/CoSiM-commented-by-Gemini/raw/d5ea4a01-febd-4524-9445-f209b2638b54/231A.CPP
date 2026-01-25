// CodeForces
// Team
// GNU C++17

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int numberOfTests = 0, validProblems = 0, count = 0;
  cin >> numberOfTests;
  while (numberOfTests > 0) {
    string guess, guessTwo, guessThree;
    cin >> guess >> guessTwo >> guessThree;
    if (guess == "1")
      count++;
    if (guessTwo == "1")
      count++;
    if (guessThree == "1")
      count++;
    if (count >= 2)
      validProblems++;
    numberOfTests--;
    count = 0;
  }
  cout << validProblems << endl;
}