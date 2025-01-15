#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  string delimiter = "+";
  string input;
  size_t pos = 0;
  string token;
  vector<int> numbers;

  cin >> input;

  /*
   * The loop finds instances of delimiter
   * Separates the token
   * Pushes the token into the vector
   * And deletes the token along with the delimiter
   */
  while ((pos = input.find(delimiter)) != string::npos) {
    token = input.substr(0, pos);
    numbers.push_back(stoi(token));
    input.erase(0, pos + delimiter.length());
  }
  // Leftover token
  numbers.push_back(stoi(input));
  sort(numbers.begin(), numbers.end());

  string output;
  for (int i : numbers) {
    output.append(to_string(i) + "+");
  }
  // Removing "+" at last
  output.erase(output.length() - 1);

  std::cout << output << endl;
}