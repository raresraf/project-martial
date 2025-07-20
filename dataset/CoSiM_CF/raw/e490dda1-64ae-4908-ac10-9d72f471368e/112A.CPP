#include <bits/stdc++.h>

using namespace std;

int main() {
  string first, second;
  int totalofFmS = 0;
  cin >> first >> second;

  for (int i = 0; i < first.size(); ++i) {
    first[i] = tolower(first[i]);
    second[i] = tolower(second[i]);

    // totalofFmS += (int)elementAtiF - (int)elementAtiS;
  }

  // if(totalofFmS < 0) totalofFmS = -1;
  // else if(totalofFmS > 0 ) totalofFmS = 1;

  cout << first.compare(second) << endl;
}