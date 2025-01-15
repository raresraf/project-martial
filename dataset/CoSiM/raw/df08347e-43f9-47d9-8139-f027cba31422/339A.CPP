#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
void extract_data(string &, vector<char> &);

int main() {
  string input;
  cin >> input;

  vector<char> data;
  // extract data and store in vector
  extract_data(input, data);

  // sort the vector
  sort(data.begin(), data.end());

  int i = 0;

  while (i < data.size()) {

    if (i != data.size() - 1)
      cout << data.at(i) << '+';
    else
      cout << data.at(i) << endl;

    i++;
  }
}

void extract_data(string &str, vector<char> &data) {
  int i = 0;

  while (i < str.length()) {
    // push the elements to vector
    if (str.at(i) == '+')
      data.push_back(str.at(i - 1));
    i++;
  }

  // push the last item
  data.push_back(str.at(str.length() - 1));
}